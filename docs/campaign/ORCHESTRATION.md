# Experiment orchestration — use scripts/exp.py

One tested entry point replaces the per-arm bash sprawl:

    PYTHONPATH=src .venv/bin/python scripts/exp.py pack  --out data/shards/pool-x A.docs.jsonl B.docs.jsonl
    PYTHONPATH=src .venv/bin/python scripts/exp.py chain --name x-96m --shards data/shards/pool-x \
        --init-region skeleton --mid-template configs/train/midtrain-staged-96m.yaml
    PYTHONPATH=src .venv/bin/python scripts/exp.py eval  --model catalog:runs/sft-x/latest.pt --out runs/evalsuite/x.json
    PYTHONPATH=src .venv/bin/python scripts/exp.py queue experiments/queue-face.txt prepend 'bash ...'

Built-in behaviours the bash copies kept reimplementing (and breaking): the face numpy
shadow, LOCALAGENT_RESUME_LINEAGE=warn, pretrain artifact stage-guards, atomic queue
rewrites, pack manifest skip. Unit tests: scripts/test_exp.py (pytest, no cluster needed).

Superseded for NEW work (keep for running/queued jobs only):
skeleton_arm.sh · staged_chain.sh · build_pool_open.sh · build_pool_general.sh ·
pack_r3_pools.sh · convert_pool.sh · seed_queues.sh · thor2_fresh_chain.sh · face_*.sh ·
run_bigpretrain.sh · eval_ladder_chain.sh · eval_mobileactions_all.sh
