import csv
import base64
import openai
import os
import re
import time
import difflib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def normalize(text):
    return text.lower().replace(".", "").replace(",", "").strip()

# --- MINI UPGRADE: helper funkcije ---
def extract_numbered_lines(text):
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^\d+\.", s):
            lines.append(s)
    return lines

def normalize_step(line):
    s = re.sub(r"^\d+\.\s*", "", line).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)  # makni zagrade
    s = s.replace(".", "").replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lcs_length(a, b):
    # a, b su liste (npr. normalizirani stringovi koraka)
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def pairwise_order_score(expected_steps, output_steps):
    """
    expected_steps/output_steps: liste NORMALIZIRANIH koraka (bez numeracije).
    Račun: broj ispravno poredanih parova / ukupan broj parova u GT (n*(n-1)/2).
    Par je ispravan ako se oba koraka pojavljuju u outputu i njihov redoslijed je isti.
    """
    n = len(expected_steps)
    if n <= 1:
        return 1.0  # trivijalno

    pos = {s: i for i, s in enumerate(output_steps)}
    correct = 0
    total_pairs = n*(n-1)//2
    for i in range(n):
        for j in range(i+1, n):
            si, sj = expected_steps[i], expected_steps[j]
            if si in pos and sj in pos and pos[si] < pos[sj]:
                correct += 1
    return correct/total_pairs if total_pairs else 0.0

def is_consecutive_numbering(output_text):
    """
    Provjera da su linije numerirane 1., 2., 3., ... bez rupa i počinju od 1.
    """
    nums = []
    for line in output_text.split("\n"):
        m = re.match(r"^(\d+)\.\s*", line.strip())
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return False
    return nums == list(range(1, len(nums)+1))
# --- kraj helper funkcija ---

prompts = {
    "spajanje": """Na temelju ove slike, koji bi bili logični koraci za sastaviti prikazani objekt?
    Koraci neka budu točno oblika:
    1. razina: Postavi [boja] blok.
    2. razina: Postavi [boja] blok.
    3. razina: Postavi [boja] blok.
    Ako su dva bloka u istoj ravnini, redoslijed neka bude s lijeva na desno.
    Ako je na slici LEGO kocka, navedi i mali opis njenog položaja u odnosu na ostale
    (npr. 'pomaknut udesno').
    Ako je blok kvadar, korak nek bude oblika: Postavi [boja] blok okomito/vodoravno."""
,
    "pospremanje": "Na temelju ove slike, koji bi bili logični koraci za pospremiti prikazane blokove? Koraci neka budu oblika: 1. Ukloni *boja* blok.",
    "prijenos": "Na temelju slike, navedi logične korake za premještanje prikazanih objekata. \
    Koraci neka budu točno oblika:\n\
    1. Premjesti *boja* objekt (*broj ljudi* osobe).\n\
    2. Premjesti *boja* objekt (*broj ljudi* osobe).\n\
    Korake navedi redoslijedom kojim bi se premještanje trebalo izvesti, \
    uzimajući u obzir veličinu i težinu objekata. \
    Ako je za objekt potreban veći broj ljudi, obavezno to naznači u zagradi. \
    Koristi samo formatirani popis koraka, bez dodatnog teksta."


}

ground_truth = []
with open("dataset/annotations.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ground_truth.append({
            "image": row["image_filename"],
            "expected": row["expected_output"],
            "category": row["category"]
        })

results = []
for idx, row in enumerate(ground_truth, start=1):
    image_name = row["image"]
    expected_output = row["expected"]
    category = row["category"]
    prompt = prompts.get(category, "Na temelju slike, navedi logične korake.")

    print(f"[INFO] Obradujem sliku {idx}/{len(ground_truth)}: {image_name} ({category})")

    image_path = os.path.join("dataset/images", image_name)
    base64_image = encode_image(image_path)

    start_time = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=500
    )

    duration = time.time() - start_time
    output = response['choices'][0]['message']['content']

    # --- MINI UPGRADE: robustno izdvajanje i normalizacija koraka ---
    expected_raw = [line for line in expected_output.split("\n") if line.strip()]
    # koristimo normalize_step za konzistentnost (prefiks "1." dodan samo da funkcija radi isto kao za output)
    expected_steps = [normalize_step(line) for line in expected_raw]

    output_numbered = extract_numbered_lines(output)
    output_steps = [normalize_step(line) for line in output_numbered]
    # --- kraj izdvajanja ---

    def is_similar(a, b, threshold=0.75):
        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

    tp = 0
    matched_expected = set()

    for out_step in output_steps:
        for i, exp_step in enumerate(expected_steps):
            if i not in matched_expected and is_similar(out_step, exp_step):
                tp += 1
                matched_expected.add(i)
                break

    fp = sum(1 for step in output_steps if step not in expected_steps)
    fn = sum(1 for step in expected_steps if step not in output_steps)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # --- MINI UPGRADE: dodatne metrike ---
    lcs_len = lcs_length(expected_steps, output_steps)
    lcs_coverage = (lcs_len / len(expected_steps)) if expected_steps else 0.0
    order_score = pairwise_order_score(expected_steps, output_steps)
    format_ok = is_consecutive_numbering(output)
    # --- kraj dodatnih metrika ---

    results.append({
        "image": image_name,
        "expected": expected_output,
        "output": output,
        "category": category,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "duration": duration,
        "lcs_coverage": lcs_coverage,
        "order_score": order_score,
        "format_ok": int(format_ok)
    })

total_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0
avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0.0

print(f"\nProsječna F1 točnost: {total_f1 * 100:.2f}%")
print(f"Prosječno trajanje upita: {avg_duration:.2f} sekundi")

for r in results:
    print(f"\nSlika: {r['image']}")
    print(f"Očekivano:\n{r['expected']}")
    print(f"Dobiveno:\n{r['output']}")
    print(f"Preciznost: {r['precision']*100:.2f}%, Opoziv: {r['recall']*100:.2f}%, F1: {r['f1']*100:.2f}%")
    print(f"LCS: {r['lcs_coverage']*100:.2f}%, Order: {r['order_score']*100:.2f}%, Format OK: {r['format_ok']}")
    print(f"Trajanje: {r['duration']:.2f} sekundi")

# ===================== NOVI BLOK ZA GRAFOVE (5 prikaza) =====================
import numpy as np
import re
from collections import defaultdict, Counter

# ------------- priprema metrika po kategoriji -------------
cat_stats = defaultdict(lambda: {
    "f1": [], "precision": [], "recall": [],
    "lcs": [], "order": [], "duration": [], "valid": []
})

for r in results:
    c = r["category"]
    cat_stats[c]["f1"].append(r["f1"])
    cat_stats[c]["precision"].append(r["precision"])
    cat_stats[c]["recall"].append(r["recall"])
    cat_stats[c]["lcs"].append(r["lcs_coverage"])
    cat_stats[c]["order"].append(r["order_score"])
    cat_stats[c]["duration"].append(r["duration"])
    cat_stats[c]["valid"].append(r["format_ok"])

categories = list(cat_stats.keys())

# Helperi za sigurno računanje prosjeka
def safe_mean(x):
    return sum(x)/len(x) if x else 0.0

# 1) BAR: Prosjek LCS i Order score po kategoriji
lcs_means = [safe_mean(cat_stats[c]["lcs"]) for c in categories]
order_means = [safe_mean(cat_stats[c]["order"]) for c in categories]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, lcs_means, width=width, label="LCS coverage")
plt.bar(x + width/2, order_means, width=width, label="Order score")
plt.xticks(x, categories)
plt.ylim(0, 1.05)
plt.xlabel("Kategorije")
plt.ylabel("Vrijednost (0–1)")
plt.title("LCS i Order score po kategoriji")
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("graf_lcs_order_po_kategoriji.png", dpi=200)
plt.show()

# 2) BOX: Distribucija trajanja upita po kategoriji
plt.figure(figsize=(10, 6))
data = [cat_stats[c]["duration"] for c in categories]
plt.boxplot(data, labels=categories, showmeans=True)
plt.ylabel("Trajanje upita (s)")
plt.title("Distribucija trajanja po kategoriji")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("graf_trajanje_boxplot_po_kategoriji.png", dpi=200)
plt.show()

# 3) SCATTER: F1 vs LCS (po slici)
f1_all = [r["f1"] for r in results]
lcs_all = [r["lcs_coverage"] for r in results]

plt.figure(figsize=(8, 6))
plt.scatter(f1_all, lcs_all)
plt.xlabel("F1")
plt.ylabel("LCS coverage")
plt.title("F1 vs LCS po slici")
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("graf_scatter_f1_vs_lcs.png", dpi=200)
plt.show()


# 5) BOX: Precision/Recall/F1 po kategoriji (tri odvojena boxplota)
metric_names = ["precision", "recall", "f1"]
metric_titles = {"precision": "Precision", "recall": "Recall", "f1": "F1"}

for m in metric_names:
    plt.figure(figsize=(10, 6))
    data = [cat_stats[c][m] for c in categories]
    plt.boxplot(data, labels=categories, showmeans=True)
    plt.ylim(-0.02, 1.02)
    plt.ylabel(metric_titles[m])
    plt.title(f"{metric_titles[m]} po kategoriji")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"graf_box_{m}_po_kategoriji.png", dpi=200)
    plt.show()


# --- GRAF: Validnost formata po kategoriji (prosjek format_ok) ---
valid_means = [safe_mean(cat_stats[c]["valid"]) for c in categories]

plt.figure(figsize=(8, 5))
plt.bar(categories, valid_means)
plt.ylim(0, 1.02)
plt.xlabel("Kategorije")
plt.ylabel("Udio valjanih odgovora")
plt.title("Validnost formata po kategoriji")
# ispisi postotke iznad stupaca
for i, v in enumerate(valid_means):
    plt.text(i, min(v + 0.03, 1.0), f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.savefig("graf_validnost_po_kategoriji.png", dpi=200)
plt.show()

# ===================== KRAJ NOVOG BLOKA =====================


with open("rezultati_output.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image", "category", "expected", "output",
        "precision", "recall", "f1", "duration",
        "lcs_coverage", "order_score", "format_ok"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "image": r["image"],
            "category": r["category"],
            "expected": r["expected"],
            "output": r["output"],
            "precision": f"{r['precision']:.4f}",
            "recall": f"{r['recall']:.4f}",
            "f1": f"{r['f1']:.4f}",
            "duration": f"{r['duration']:.2f}",
            "lcs_coverage": f"{r['lcs_coverage']:.4f}",
            "order_score": f"{r['order_score']:.4f}",
            "format_ok": r["format_ok"]
        })

from collections import defaultdict
import statistics

# grupiranje po kategoriji
cat_groups = defaultdict(list)
for r in results:
    cat_groups[r["category"]].append(r)

# ispis zaglavlja tablice
print("\n=== Sažetak metrika po kategorijama ===")
print(f"{'Kategorija':<15}{'Uzoraka':<10}{'Prec.':<8}{'Recall':<8}{'F1':<8}{'LCS':<8}{'Order':<8}{'Valid%':<8}{'Trajanje(s)':<12}")

# prolazak po kategorijama
for cat, items in cat_groups.items():
    samples = len(items)
    avg_precision = statistics.mean(r["precision"] for r in items)
    avg_recall = statistics.mean(r["recall"] for r in items)
    avg_f1 = statistics.mean(r["f1"] for r in items)
    avg_lcs = statistics.mean(r["lcs_coverage"] for r in items)
    avg_order = statistics.mean(r["order_score"] for r in items)
    avg_duration = statistics.mean(r["duration"] for r in items)
    avg_valid = statistics.mean(r["format_ok"] for r in items)
    
    print(f"{cat:<15}{samples:<10}{avg_precision:.3f}  {avg_recall:.3f}  {avg_f1:.3f}  {avg_lcs:.3f}  {avg_order:.3f} {avg_valid*100:>6.1f} {avg_duration:.3f}")

# spremanje u CSV ručno
with open("summary_by_category.csv", "w", encoding="utf-8") as f:
    f.write("category,samples,avg_precision,avg_recall,avg_f1,avg_lcs,avg_order,avg_duration\n")
    for cat, items in cat_groups.items():
        samples = len(items)
        avg_precision = statistics.mean(r["precision"] for r in items)
        avg_recall = statistics.mean(r["recall"] for r in items)
        avg_f1 = statistics.mean(r["f1"] for r in items)
        avg_lcs = statistics.mean(r["lcs_coverage"] for r in items)
        avg_order = statistics.mean(r["order_score"] for r in items)
        avg_duration = statistics.mean(r["duration"] for r in items)
        avg_valid = statistics.mean(r["format_ok"] for r in items)  
        f.write(f"{cat},{samples},{avg_precision:.3f},{avg_recall:.3f},{avg_f1:.3f},{avg_lcs:.3f},{avg_order:.3f},{avg_valid:.3f},{avg_duration:.3f}\n")

print("\nTablica spremljena kao summary_by_category.csv")
