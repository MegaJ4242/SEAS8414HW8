import json, re
from pathlib import Path
import pandas as pd
from pycaret.clustering import load_model, predict_model

FEATURES = [
    "SSLfinal_State","Prefix_Suffix","Shortining_Service","having_IP_Address",
    "abnormal_url","has_political_keyword","suspicious_tld","url_length",
    "num_subdomains","redirect_count","domain_age_days","https_token_in_url",
]

# load data and malicious subset
df = pd.read_csv("data/train.csv")
df_mal = df[df["label"]=="MALICIOUS"].copy()

# load clusterer and assign clusters
clu = load_model("models/threat_actor_profiler")
pred = predict_model(clu, data=df_mal[FEATURES])

# normalize cluster id (handles '0' or 'Cluster 0')
if "Cluster" in pred.columns:
    s = pred["Cluster"].astype(str).str.extract(r"(\d+)")
else:
    # fallback: first int-looking column
    s = None
    for c in pred.columns:
        if pred[c].astype(str).str.match(r"^\d+$").any():
            s = pred[c].astype(str).str.extract(r"(\d+)")
            break
    if s is None:
        raise RuntimeError("Could not find cluster column in predict_model output.")

pred["_cluster_id"] = s[0].astype(int)

# compute centroids in raw feature space
centroids = pred.groupby("_cluster_id")[FEATURES].mean()

def score_state(c):
    return (1.5*c["SSLfinal_State"] + 1.0*c["Prefix_Suffix"] + 0.8*(c["domain_age_days"]/4000.0)
            - 1.0*c["Shortining_Service"] - 1.0*c["having_IP_Address"]
            - 0.5*c["suspicious_tld"] - 0.5*(c["redirect_count"]/10.0))

def score_crime(c):
    return (1.2*c["Shortining_Service"] + 1.2*c["having_IP_Address"] + 1.0*c["abnormal_url"]
            + 1.0*c["suspicious_tld"] + 0.7*(c["url_length"]/200.0)
            + 0.7*(c["num_subdomains"]/5.0) + 0.7*(c["redirect_count"]/10.0)
            - 0.5*c["SSLfinal_State"])

def score_hacktivist(c):
    return (1.5*c["has_political_keyword"] + 0.8*c["abnormal_url"]
            + 0.3*c["SSLfinal_State"] + 0.2*(c["domain_age_days"]/4000.0))

# greedy unique assignment
triples = []
for cid in centroids.index:
    c = centroids.loc[cid]
    triples += [
        (int(cid), "State-Sponsored", float(score_state(c))),
        (int(cid), "Organized Cybercrime", float(score_crime(c))),
        (int(cid), "Hacktivist", float(score_hacktivist(c))),
    ]
triples.sort(key=lambda t: t[2], reverse=True)

mapping, used, remaining = {}, set(), {"State-Sponsored","Organized Cybercrime","Hacktivist"}
for cid, profile, _ in triples:
    if cid not in used and profile in remaining:
        mapping[cid] = profile
        used.add(cid); remaining.remove(profile)
    if len(mapping) == 3:
        break

Path("models").mkdir(exist_ok=True, parents=True)
with open("models/cluster_profile_map.json","w") as f:
    json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

print("Saved mapping:", mapping)
