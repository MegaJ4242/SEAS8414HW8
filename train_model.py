\
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pycaret.classification import (compare_models, finalize_model, plot_model,
                                    save_model)
from pycaret.classification import setup as cls_setup
from pycaret.clustering import assign_model as clu_assign_model
from pycaret.clustering import create_model as clu_create_model
from pycaret.clustering import plot_model as clu_plot_model
from pycaret.clustering import save_model as clu_save_model
from pycaret.clustering import setup as clu_setup

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def ensure_dirs():
    Path("models").mkdir(exist_ok=True, parents=True)
    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("data").mkdir(exist_ok=True, parents=True)


def generate_synthetic_data(n_rows=6000, benign_ratio=0.4):
    n_benign = int(n_rows * benign_ratio)
    n_mal = n_rows - n_benign
    n_each = n_mal // 3
    n_state, n_crime, n_hackt = n_each, n_each, n_mal - 2 * n_each
    rows = []
    for _ in range(n_benign):
        rows.append(dict(SSLfinal_State=1, Prefix_Suffix=np.random.binomial(1, 0.05),
                         Shortining_Service=np.random.binomial(1, 0.02), having_IP_Address=np.random.binomial(1, 0.01),
                         abnormal_url=np.random.binomial(1, 0.05), has_political_keyword=np.random.binomial(1, 0.01),
                         suspicious_tld=np.random.binomial(1, 0.03), url_length=int(np.random.normal(30, 8)),
                         num_subdomains=int(np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])),
                         redirect_count=int(np.random.choice([0, 1], p=[0.85, 0.15])),
                         domain_age_days=int(np.random.normal(2200, 400)), https_token_in_url=np.random.binomial(1, 0.02),
                         label="BENIGN", actor_profile="BENIGN"))
    for _ in range(n_state):
        rows.append(dict(SSLfinal_State=1, Prefix_Suffix=np.random.binomial(1, 0.6),
                         Shortining_Service=0, having_IP_Address=0, abnormal_url=np.random.binomial(1, 0.25),
                         has_political_keyword=0, suspicious_tld=np.random.binomial(1, 0.1),
                         url_length=int(np.random.normal(55, 10)), num_subdomains=int(np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])),
                         redirect_count=int(np.random.choice([0, 1], p=[0.9, 0.1])), domain_age_days=int(np.random.normal(2000, 600)),
                         https_token_in_url=0, label="MALICIOUS", actor_profile="State-Sponsored"))
    for _ in range(n_crime):
        rows.append(dict(SSLfinal_State=np.random.binomial(1, 0.2), Prefix_Suffix=np.random.binomial(1, 0.7),
                         Shortining_Service=1, having_IP_Address=1, abnormal_url=1, has_political_keyword=0,
                         suspicious_tld=1, url_length=int(np.random.normal(90, 15)),
                         num_subdomains=int(np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.2])),
                         redirect_count=int(np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])),
                         domain_age_days=int(np.random.normal(60, 30)), https_token_in_url=np.random.binomial(1, 0.6),
                         label="MALICIOUS", actor_profile="Organized Cybercrime"))
    for _ in range(n_hackt):
        rows.append(dict(SSLfinal_State=np.random.binomial(1, 0.7), Prefix_Suffix=np.random.binomial(1, 0.3),
                         Shortining_Service=np.random.binomial(1, 0.15), having_IP_Address=np.random.binomial(1, 0.05),
                         abnormal_url=np.random.binomial(1, 0.5), has_political_keyword=1,
                         suspicious_tld=np.random.binomial(1, 0.2), url_length=int(np.random.normal(45, 12)),
                         num_subdomains=int(np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])),
                         redirect_count=int(np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])),
                         domain_age_days=int(np.random.normal(800, 300)), https_token_in_url=np.random.binomial(1, 0.2),
                         label="MALICIOUS", actor_profile="Hacktivist"))
    df = pd.DataFrame(rows)
    df["url_length"] = df["url_length"].clip(5, 200)
    df["num_subdomains"] = df["num_subdomains"].clip(0, 5)
    df["redirect_count"] = df["redirect_count"].clip(0, 10)
    df["domain_age_days"] = df["domain_age_days"].clip(1, 4000)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def save_plot_safely(fn):
    import glob
    import shutil
    imgs = sorted(glob.glob("*.png"), key=os.path.getmtime)
    if imgs:
        shutil.move(imgs[-1], f"reports/{fn}")


def train_classifier(df: pd.DataFrame, feature_cols):
    print("[*] Training classification model...")
    df_cls = df[feature_cols + ["label"]].copy()
    cls_setup(data=df_cls, target="label", session_id=RANDOM_STATE, fold=5, verbose=False)
    best = compare_models(include=["lr", "rf", "dt", "knn", "nb", "svm", "ridge", "gbc"])
    final_best = finalize_model(best)
    save_model(final_best, "models/phishing_url_detector")
    plot_model(final_best, plot="confusion_matrix", save=True)
    save_plot_safely("confusion_matrix.png")
    plot_model(final_best, plot="auc", save=True)
    save_plot_safely("roc.png")
    plot_model(final_best, plot="pr", save=True)
    save_plot_safely("pr.png")
    plot_model(final_best, plot="feature", save=True)
    save_plot_safely("feature.png")
    print("[+] Classifier trained and saved to models/phishing_url_detector.pkl")
    return final_best


def train_clusterer(df_malicious, feature_cols):
    clu_setup(data=df_malicious[feature_cols], session_id=RANDOM_STATE, normalize=True, verbose=False)
    kmeans = clu_create_model("kmeans", num_clusters=3, random_state=RANDOM_STATE)
    labeled = clu_assign_model(kmeans)
    clu_save_model(kmeans, "models/threat_actor_profiler")
    try:
        clu_plot_model(kmeans, plot="elbow", save=True)
        save_plot_safely("elbow.png")
        clu_plot_model(kmeans, plot="silhouette", save=True)
        save_plot_safely("silhouette.png")
        clu_plot_model(kmeans, plot="cluster", save=True)
        save_plot_safely("cluster.png")
    except Exception as e:
        print(f"[!] Plot error: {e}")

    # Map clusters -> profiles using centroid heuristics
    centroids = labeled.groupby("Cluster")[feature_cols].mean()

    def s_state(c): return 1.5 * c["SSLfinal_State"] + 1.0 * c["Prefix_Suffix"] + 0.8 * (c["domain_age_days"] / 4000.0) - 1.0 * \
        c["Shortining_Service"] - 1.0 * c["having_IP_Address"] - 0.5 * c["suspicious_tld"] - 0.5 * c["redirect_count"] / 10.0

    def s_crime(c): return 1.2 * c["Shortining_Service"] + 1.2 * c["having_IP_Address"] + 1.0 * c["abnormal_url"] + 1.0 * c["suspicious_tld"] + 0.7 * (
        c["url_length"] / 200.0) + 0.7 * (c["num_subdomains"] / 5.0) + 0.7 * (c["redirect_count"] / 10.0) - 0.5 * c["SSLfinal_State"]
    def s_hackt(c): return 1.5 * c["has_political_keyword"] + 0.8 * c["abnormal_url"] + \
        0.3 * c["SSLfinal_State"] + 0.2 * (c["domain_age_days"] / 4000.0)

    import pandas as pd
    scores = []
    for cl in centroids.index:
        c = centroids.loc[cl]
        scores.append((int(cl), float(s_state(c)), float(s_crime(c)), float(s_hackt(c))))
    df_scores = pd.DataFrame(scores, columns=["Cluster", "State", "Crime", "Hackt"]).set_index("Cluster")

    cluster_map, remaining = {}, {"State-Sponsored", "Organized Cybercrime", "Hacktivist"}
    triples = []
    for cl, r in df_scores.iterrows():
        triples += [(cl, "State-Sponsored", r["State"]),
                    (cl, "Organized Cybercrime", r["Crime"]), (cl, "Hacktivist", r["Hackt"])]
    triples.sort(key=lambda t: t[2], reverse=True)
    used = set()
    for cl, profile, _ in triples:
        if cl not in used and profile in remaining:
            cluster_map[cl] = profile
            used.add(cl)
            remaining.remove(profile)
        if len(cluster_map) == 3:
            break

    with open("models/cluster_profile_map.json", "w") as f:
        json.dump({str(k): v for k, v in cluster_map.items()}, f, indent=2)


def main():
    ensure_dirs()
    df = generate_synthetic_data()
    df.to_csv("data/train.csv", index=False)
    feature_cols = [
        "SSLfinal_State",
        "Prefix_Suffix",
        "Shortining_Service",
        "having_IP_Address",
        "abnormal_url",
        "has_political_keyword",
        "suspicious_tld",
        "url_length",
        "num_subdomains",
        "redirect_count",
        "domain_age_days",
        "https_token_in_url"]
    train_classifier(df, feature_cols)
    train_clusterer(df[df["label"] == "MALICIOUS"], feature_cols)
    print("[✓] Training complete: models in ./models, plots in ./reports")


if __name__ == "__main__":
    main()
