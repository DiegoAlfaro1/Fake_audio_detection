from sklearn.metrics import roc_auc_score   

MODEL_PATH = "./training_checkpoints/checkpoint_16.keras"   

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PerSampleStandardization": PerSampleStandardization},
)

test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
print("Compiled test metrics:", {k: round(float(v), 4) for k, v in test_metrics.items()})

y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print(f"\nAUC (sklearn): {roc_auc_score(y_true, y_prob):.4f}")
fpr, tpr, thr = roc_curve(y_true, y_prob)
fnr = 1 - tpr
i = np.nanargmin(np.abs(fnr - fpr))
print(f"EER: {(fpr[i] + fnr[i]) / 2:.4f} at threshold {thr[i]:.4f}")

print("\n--- threshold = 0.50 ---")
print(classification_report(y_true, y_pred, target_names=["fake (0)", "real (1)"], digits=4))
print(f"Macro F1 @ 0.50: {f1_score(y_true, y_pred, average='macro'):.4f}")

val_true = np.concatenate([y.numpy() for _, y in val_ds]).astype(int)
val_prob = model.predict(val_ds, verbose=0).ravel()
ths = np.linspace(0.0, 1.0, 1001)
op_t = ths[int(np.argmax([f1_score(val_true, (val_prob >= t).astype(int), average="macro") for t in ths]))]
y_pred_op = (y_prob >= op_t).astype(int)
print(f"\n--- threshold = {op_t:.4f} (chosen on val) ---")
print(classification_report(y_true, y_pred_op, target_names=["fake (0)", "real (1)"], digits=4))
print(f"Macro F1 @ val threshold: {f1_score(y_true, y_pred_op, average='macro'):.4f}")

cm = confusion_matrix(y_true, y_pred_op)
ConfusionMatrixDisplay(cm, display_labels=["fake", "real"]).plot(cmap="Blues", values_format="d")
plt.title(f"Confusion matrix (test) @ threshold {op_t:.3f}")
plt.show()