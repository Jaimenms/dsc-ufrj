import numpy as np, time, sys, traceback, os
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators import integrate_implicit

files = sys.argv[1:] or [
    "CASO10-1-1",
    "CASO10-1-2",
    "CASO10-1-3",
    "CASO10-1-4",
    "CASO10-2-1",
    "CASO10-2-1b",
    "CASO10-2-2",
    "CASO10-3-1",
    "CASO10-3-2",
    "CASO10-3-2b",
    "CASO10-3-3b",
]
for f in files:
    print("=" * 70)
    print(f)
    try:
        case, raw = load_case(f + ".mf2")
        model = build_model(case)
        Tl, Xl = legacy_states(raw)
        print(" method", case.pipes[0].scheme.type, "n_states", model.n_states, "legacy", Xl.shape, "tf", Tl[-1])
        X0 = Xl[:, 0]
        r0 = model.rhs(Tl[0], X0)
        print(f" |rhs(X0)|={np.linalg.norm(r0):.3e}")
        t0 = time.time()
        T, X, info = integrate_implicit(model, X0, Tl, rtol=1e-4, atol=1e-6, bounds=state_bounds(model), progress=False)
        print(
            f" implicit: steps={info['n_steps']} rej={info['n_rejected']} rhs={info['n_rhs']} jac={info['n_jac']} {time.time()-t0:.1f}s"
        )
        for k, pb in enumerate(model.pipes):
            slots = pb.free % 4
            seg = slice(pb.offset, pb.offset + pb.n_states)
            msg = []
            for s, name in enumerate(["aG", "PT", "qL", "qG"]):
                m = slots == s
                if m.any():
                    d = np.max(np.abs(X[seg][m] - Xl[seg][m]))
                    rng = np.ptp(Xl[seg][m])
                    msg.append(f"{name}:{d:.2e}/{rng:.2e}")
            print(f"  pipe{k+1} ({pb.model.name}) maxdiff/range:", " ".join(msg))
        if model.vA:
            seg = slice(model.n_pipe_states, None)
            print("  vertA maxdiff:", np.max(np.abs(X[seg] - Xl[seg]), axis=1))
        os.makedirs("runs", exist_ok=True)
        np.save(f"runs/X_{f}_implicit.npy", X)
    except Exception as e:
        traceback.print_exc()
