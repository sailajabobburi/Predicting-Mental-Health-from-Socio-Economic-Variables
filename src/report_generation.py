def generate_report(performances):
    report = []
    for perf in performances:
        model_name = perf['model'].__class__.__name__
        accuracy = perf['accuracy']
        report.append(f"Model: {model_name}")
        report.append(f"Accuracy: {accuracy:.4f}")
        report.append("Confusion Matrix:")
        report.append(str(perf['conf_matrix']))
        report.append("Classification Report:")
        report.append(str(perf['class_report']))
        report.append("\n" + "=" * 50 + "\n")

    return "\n".join(report)
