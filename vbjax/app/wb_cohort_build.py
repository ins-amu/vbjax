"""Build a cohort of subjects from the 1000Brains Schaefer-100 archive.

The archive (~/mhsa_e0/data/kb_extract/100-Schaefer17Networks.zip) holds, per
subject and session:
    SC/<id>_<sess>_Counts.csv    100x100 streamline counts
    SC/<id>_<sess>_Lengths.csv   100x100 mean tract lengths
    BOLD/<id>_<sess>_RestEmpBOLD.csv   T x 100 resting BOLD (300 rows)

We take session _1 for every subject that has one, truncate BOLD to the shortest
T, and save arrays so the cohort fit does not re-parse the archive:
    bold_all.npy    (K, T, N)
    counts_all.npy  (K, N, N)
    lengths_all.npy (K, N, N)
    subj_ids.npy    (K,)

Run (CPU is fine):
  python -m vbjax.app.wb_cohort_build --zip ... --out ~/mhsa_e0/data/kb_cohort
"""
import argparse, os, re, zipfile
import numpy as onp


def parse_names(zf):
    bold, counts, lengths = {}, {}, {}
    for n in zf.namelist():
        if not n.endswith(".csv"):
            continue
        m = re.search(r"/(\d+)_(\d+)_(RestEmpBOLD|Counts|Lengths)\.csv$", n)
        if not m:
            continue
        sid, sess, kind = m.group(1), m.group(2), m.group(3)
        key = (sid, sess)
        {"RestEmpBOLD": bold, "Counts": counts, "Lengths": lengths}[kind][key] = n
    return bold, counts, lengths


def load_csv(zf, name):
    with zf.open(name) as fh:
        return onp.loadtxt(fh, delimiter=",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default="1")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        bold, counts, lengths = parse_names(zf)
        keys = sorted(k for k in bold if k[1] == args.session
                      and k in counts and k in lengths)
        print(f"[build] {len(bold)} BOLD files, {len(keys)} subjects with "
              f"session _{args.session} SC+BOLD", flush=True)

        rows, cs, ls, ids = [], [], [], []
        for sid, sess in keys:
            b = load_csv(zf, bold[(sid, sess)])
            c = load_csv(zf, counts[(sid, sess)])
            l = load_csv(zf, lengths[(sid, sess)])
            if b.ndim != 2 or b.shape[1] != 100 or c.shape != (100, 100):
                print(f"[build] skip {sid}: shapes b={b.shape} c={c.shape}", flush=True)
                continue
            rows.append(b.astype("f4")); cs.append(c.astype("f4")); ls.append(l.astype("f4"))
            ids.append(int(sid))

    T = min(r.shape[0] for r in rows)
    bold_all = onp.stack([r[:T] for r in rows])
    os.makedirs(args.out, exist_ok=True)
    onp.save(os.path.join(args.out, "bold_all.npy"), bold_all)
    onp.save(os.path.join(args.out, "counts_all.npy"), onp.stack(cs))
    onp.save(os.path.join(args.out, "lengths_all.npy"), onp.stack(ls))
    onp.save(os.path.join(args.out, "subj_ids.npy"), onp.asarray(ids))
    print(f"[build] K={bold_all.shape[0]} T={T} N={bold_all.shape[2]} -> {args.out}",
          flush=True)
    print("WB_COHORT_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()