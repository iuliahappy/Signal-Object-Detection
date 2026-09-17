import os
import numpy as np
import pandas as pd

BASE_PATH = '.'
OUT_PATH = os.getcwd()
NUM_CLASSES = 5

PROB_FILES = [
    (os.path.join(OUT_PATH, 'test_probs.npy'),       1.0),   # 128x64
    (os.path.join(OUT_PATH, 'test_probs_hires.npy'), 1.0),   # 160x96
    (os.path.join(OUT_PATH, 'test_probs_seed2.npy'),  1.0),  # seed 123
]

def main():
    test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    n = len(test_df)

    blended = np.zeros((n, NUM_CLASSES))
    wsum = 0.0
    used = []
    per_model_preds = {}

    for path, w in PROB_FILES:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
            
        p = np.load(path)
        
        if p.shape != (n, NUM_CLASSES):
            print(f"Wrong shape for {path}: expected {(n, NUM_CLASSES)}, got {p.shape}. Skip!")
            continue
            
        # normalize each model to sum 1 per row 
        p = p / p.sum(axis=1, keepdims=True).clip(1e-9)
        blended += w * p
        wsum += w
        used.append((os.path.basename(path), w))
        per_model_preds[os.path.basename(path)] = p.argmax(1)

    if wsum <= 0:
        print("No valid probability files were loaded! Exiting.")
        return

    blended /= wsum

    print("\nModels used in blend:")
    for name, w in used:
        print(f"{name} (weight: {w})")

    # checking how much the models differ (diversity = ensemble potential)
    names = list(per_model_preds.keys())
    if len(names) >= 2:
        print("\nModel agreement (percentage of identical predictions):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agree = (per_model_preds[names[i]] == per_model_preds[names[j]]).mean() * 100
                print(f"   {names[i]} vs {names[j]}: {agree:.1f}%")

    final_preds = blended.argmax(axis=1) + 1
    sub = pd.DataFrame({'id': test_df['id'], 'label': final_preds})
    out_file = os.path.join(OUT_PATH, 'sample_submission.csv')
    sub.to_csv(out_file, index=False)

    print("\nFinal prediction distribution (Blend):")
    counts = sub['label'].value_counts().sort_index()
    for c, count in counts.items():
        print(f"Class {c}: {count:4d} ({count / n * 100:.1f}%)")
    print(f"\nSaved: {out_file}")

if __name__ == '__main__':
    main()