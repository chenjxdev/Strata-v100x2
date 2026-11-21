"""The engine's console figures (STATS -> XSLOT / XMAP lines, serve/engine/console.py) and what /metrics and /slots show of
them: the expert map, each slot's expert split and draft acceptance, and the expert tiers' occupancy (Lapilli).

The lines are the engine's own format (src/program/serve/console_stats.hpp); STATS goes only to an engine that says it
reads it (INFO console_stats=1, which the real engine prints after READY), once per process."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.console import DECAY, ConsoleLines, expert_tiers, parse_xmap, parse_xslot
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine, info_facts
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[2] / "serve" / "chat_template.jinja"

# a fake engine: INFO console_stats=1 after READY (as the deepseek41 engine), every stdin line kept in a file
FAKE = r'''
import os, sys
said = os.environ["FAKE_SAID"]
console = os.environ.get("FAKE_CONSOLE") == "1"
print("INFO engine=fake", flush=True)
print("READY 4096 stop", flush=True)
if console:
    print("INFO console_stats=1 eos_ids=257", flush=True)
on = False
for raw in sys.stdin:
    with open(said, "a", encoding="utf-8") as f:
        f.write(raw)
    if raw.strip() == "STATS":
        on = True
        print("XMAP 6 001233 0,0,0,0,0,0", flush=True)
    elif raw.startswith("GEN"):
        print("T 65", flush=True)
        if on:
            print("XSLOT 0 2 3 30 40 8 4 2/2,1/2", flush=True)
            print("XMAP 6 001233 4,0,2,0,1,9", flush=True)
        print("DONE 1 3 1.0 1.0 length 0 0 0", flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Lines(unittest.TestCase):
    def test_xslot_and_xmap_as_the_engine_writes_them(self):
        b, c = parse_xslot("XSLOT 1 12 30 300 400 80 40 9/10,5/10,1/6\n")
        self.assertEqual(b, 1)
        self.assertEqual(c, {"windows": 12, "tokens": 30, "hits": 300, "lookups": 400, "offloaded": 80, "files": 40,
                             "accept": [(9, 10), (5, 10), (1, 6)]})
        self.assertEqual(parse_xslot("XSLOT 0 0 0 0 0 0 0 -")[1]["accept"], [])     # before any draft
        self.assertEqual(parse_xmap("XMAP 7680 0123 5,0,1,2\n"), (7680, [0, 1, 2, 3], [5, 0, 1, 2]))
        self.assertIsNone(parse_xmap("XMAP 7680 0123 5,0,1"))                       # one count per expert or nothing
        self.assertIsNone(parse_xslot("XSLOT 0 1 2"))
        self.assertIsNone(parse_xmap("XMAP x 0 0"))

    def test_a_slot_split_and_its_rates(self):
        clk = Clock()
        con = ConsoleLines(clock=clk)
        self.assertEqual(con.slot(0), {"accept_pos": None, "accept_hist": None, "hit": None})
        con.feed("XSLOT 0 1 2 30 40 8 4 1/2")
        clk.t += 1.0
        con.feed("XSLOT 0 3 8 90 120 24 12 3/4,1/2")
        s = con.slot(0)
        # routed = lookups + offloaded = 144: VRAM 90, the other card 24, the CPU 30
        self.assertEqual(s["hit"]["vram"], round(90 / 144, 3))
        self.assertEqual(s["hit"]["peer"], round(24 / 144, 3))
        self.assertEqual(s["hit"]["cpu"], round(30 / 144, 3))
        self.assertEqual(s["hit"]["lookups_s"], 80 + 16)          # the second line's deltas over the second between them
        self.assertEqual(s["hit"]["peer_s"], 16)
        self.assertEqual(s["hit"]["ram_s"], 20)
        self.assertEqual(s["accept_pos"], [0.75, 0.5])
        self.assertEqual((s["drafts_accepted"], s["drafts_offered"]), (4, 6))
        clk.t += 3.0                                              # no line for longer than STALE_S: no window runs
        self.assertEqual(con.slot(0)["hit"]["lookups_s"], 0)
        con.feed("XSLOT 0 1 1 1 1 0 0 -")                         # a new engine run starts from zero
        self.assertEqual(con.slot(0)["hit"]["vram"], 1.0)
        self.assertIsNone(con.slot(0)["accept_pos"])

    def test_the_map_rates_and_the_fading_count(self):
        clk = Clock()
        con = ConsoleLines(clock=clk)
        self.assertIsNone(con.expert_map())
        con.feed("XMAP 4 0123 0,0,0,0")
        clk.t += 0.5
        con.feed("XMAP 4 0123 10,0,4,0")
        clk.t += 0.5
        con.feed("XMAP 4 0123 10,2,0,0")
        m = con.expert_map()
        self.assertEqual(m["total"], 4)
        self.assertEqual(m["tier"], [0, 1, 2, 3])
        self.assertEqual(m["routed"], [20.0, 2.0, 4.0, 0])        # over the last second of lines
        self.assertAlmostEqual(m["recent"][0], 10 * DECAY ** 0.5 + 10, places=1)
        clk.t += 2.0
        self.assertEqual(con.expert_map()["routed"], [20.0, 2.0, 4.0, 0])   # not stale yet
        clk.t += 1.0
        self.assertEqual(con.expert_map()["routed"], [0, 0, 0, 0])          # stale: no window runs
        self.assertLess(con.expert_map()["recent"][0], m["recent"][0])

    def test_the_summary_is_the_map_summed_by_home(self):
        clk = Clock()
        con = ConsoleLines(clock=clk)
        self.assertIsNone(con.summary())
        for routed in ("0,0,0,0,0,0", "3,1,0,2,5,4", "1,0,2,2,0,7"):
            con.feed(f"XMAP 6 001233 {routed}")
            clk.t += 0.5
        m, s = con.expert_map(), con.summary()
        par = [0.0, 0.0, 0.0, 0.0]
        for h, r in zip(m["tier"], m["routed"]):
            par[h] += r
        self.assertEqual(s["lookups_s"], [round(v, 2) for v in par])
        self.assertEqual((s["total"], s["cells"], s["kept"]), (6, 6, [2, 1, 1, 2]))
        clk.t += 3.0                                              # stale: the same zeros as the map
        self.assertEqual(con.summary()["lookups_s"], [0, 0, 0, 0])

    def test_the_tiers_as_the_engine_says_them(self):
        info = info_facts("INFO expert_slots=3120 kv=fp8+fp4 expert_cache_mib=8190 "
                          "expert_tiers=main/78/8190,peer/50/5250,ram/64/6720,file/0/0 expert_layers=40 "
                          "experts_per_layer=192 main_copies=0")
        t = expert_tiers(info)
        self.assertEqual(t["layers"], 40)
        self.assertEqual(t["per_layer"], 192)
        self.assertFalse(t["main_copies"])
        self.assertEqual(t["tiers"]["main"], {"per_layer": 78, "experts": 3120, "bytes": 8190 << 20})
        self.assertEqual(t["tiers"]["file"]["experts"], 0)
        self.assertIsNone(expert_tiers({}))                                 # a dense model or another engine
        self.assertIsNone(expert_tiers({**info, "expert_tiers": "main/1/2"}))   # every tier or nothing


class Stats(unittest.TestCase):
    def run_fake(self, console: bool, wait: bool = True):
        d = tempfile.mkdtemp()
        script, said, log = Path(d) / "fake_engine.py", Path(d) / "said.txt", Path(d) / "engine.log"
        script.write_text(FAKE, encoding="utf-8")
        real = subprocess.Popen
        env = {**os.environ, "FAKE_SAID": str(said), "FAKE_CONSOLE": "1" if console else "0"}
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **{**kw, "env": env})):
            engine = ProcessEngine("basalt", [], log=str(log))
            self.addCleanup(engine.close)
            deadline = time.time() + 5
            while console and wait and engine.console.expert_map() is None and time.time() < deadline:
                time.sleep(0.02)
            out = [t for t in engine.generate([1, 2, 3], 4, {}, threading.Event()) if t is not None]
            if not wait:
                time.sleep(0.3)                                   # a late STATS would be written by now
        engine.close()
        return engine, out, said.read_text(encoding="utf-8").splitlines(), log.read_text(encoding="utf-8")

    def test_an_engine_that_reads_stats_is_asked_once_and_its_lines_reach_the_console(self):
        engine, out, said, log = self.run_fake(True)
        self.assertEqual(out, [65])                                   # the reply is untouched by the console's lines
        self.assertEqual(said.count("STATS"), 1, said)
        self.assertLess(said.index("STATS"), next(i for i, x in enumerate(said) if x.startswith("GEN")))
        m = engine.console.expert_map()
        self.assertEqual(m["tier"], [0, 0, 1, 2, 3, 3])
        self.assertEqual(engine.console.slots[0]["lookups"], 40)
        self.assertNotIn("XMAP", log)                                 # protocol lines stay out of the engine log
        self.assertNotIn("XSLOT", log)

    def test_basalt_console_stats_0_sends_none_to_an_engine_that_reads_it(self):   # the A/B of what the figures cost
        with mock.patch.dict(os.environ, {"BASALT_CONSOLE_STATS": "0"}):
            engine, out, said, _ = self.run_fake(True, wait=False)
        self.assertEqual(out, [65])
        self.assertNotIn("STATS", said)
        self.assertTrue(any(x.startswith("GEN") for x in said), said)
        self.assertIsNone(engine.console.expert_map())

    def test_an_engine_that_does_not_say_it_is_never_sent_stats(self):
        engine, out, said, _ = self.run_fake(False)
        self.assertEqual(out, [65])
        self.assertNotIn("STATS", said)
        self.assertIsNone(engine.console.expert_map())


class Views(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "Hi.")
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(TEMPLATE))

    def test_without_console_figures_the_fields_are_there_and_empty(self):
        m = self.svc.metrics(full_map=True)
        self.assertIsNone(m["experts_map"])
        self.assertIsNone(m["experts_live"])
        self.assertIsNone(m["experts"])
        row = self.svc.slot_table()[0][0]
        for k in ("accept_pos", "accept_hist", "hit", "started", "path"):
            self.assertIn(k, row)
            self.assertIsNone(row[k])

    def test_the_map_the_tiers_and_a_slot_split(self):
        con = ConsoleLines()
        con.feed("XMAP 6 001233 4,0,2,0,1,9")
        con.feed("XSLOT 0 2 3 30 40 8 4 2/2,1/2")
        self.engine.console = con
        self.engine.info = info_facts("INFO expert_tiers=main/2/20,peer/1/10,ram/1/10,file/2/20 expert_layers=1 "
                                      "experts_per_layer=6 main_copies=0 expert_cache_mib=20")
        self.assertIsNone(self.svc.metrics()["experts_map"])            # the arrays only when asked (?experts=full)
        light = self.svc.metrics()["experts_live"]
        self.assertEqual((light["total"], light["cells"], light["homes"]), (6, 6, [2, 1, 1, 2]))
        self.assertEqual(light["lookups_s"], [4.0, 2.0, 0.0, 10.0])
        m = self.svc.metrics(full_map=True)
        self.assertEqual(m["experts_map"]["tier"], [0, 0, 1, 2, 3, 3])
        self.assertEqual(m["experts"]["tiers"]["file"]["experts"], 2)
        self.assertEqual(m["engine"]["expert_cache_mib"], 20)
        row = self.svc.slot_table()[0][0]
        self.assertEqual(row["hit"]["vram"], round(30 / 48, 3))
        self.assertEqual(row["accept_pos"], [1.0, 0.5])
        # the history's lookups a second by home (homes 0 0 1 2 3 3, one line over a second)
        extra = self.svc.telemetry_extra()
        self.assertEqual({k: extra.get(k) for k in ("experts_vram_s", "experts_peer_s", "experts_ram_s", "experts_file_s")},
                         {"experts_vram_s": 4.0, "experts_peer_s": 2.0, "experts_ram_s": 0.0, "experts_file_s": 10.0})

    def test_the_tiers_of_a_pool_say_their_totals(self):
        info = info_facts("INFO expert_tiers_total=main/7311/21853,peer/0/0,ram/9000/26000,file/8265/24000 expert_layers=48 "
                          "experts_per_layer=512")
        t = expert_tiers(info)
        self.assertTrue(t["pool"])
        self.assertEqual(t["tiers"]["main"], {"per_layer": None, "experts": 7311, "bytes": 21853 << 20})
        self.assertEqual((t["layers"], t["per_layer"]), (48, 512))
        self.assertFalse(expert_tiers(info_facts("INFO expert_tiers=main/68/22304,peer/95/31160,ram/97/31816,file/0/0 "
                                                 "expert_layers=40 experts_per_layer=192"))["pool"])

    def test_a_sampled_map_scales_its_lookups_to_all_the_experts(self):
        con = ConsoleLines()
        con.feed("XMAP 600 0123 1,2,3,4")                        # 4 of 600 experts sampled
        self.engine.console = con
        self.assertEqual(self.svc._expert_rates(), {"experts_vram_s": 150.0, "experts_peer_s": 300.0,
                                                    "experts_ram_s": 450.0, "experts_file_s": 600.0})


if __name__ == "__main__":
    unittest.main()
