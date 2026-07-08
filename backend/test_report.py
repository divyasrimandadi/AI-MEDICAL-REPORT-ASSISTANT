from backend.services.report_generator import generate_medical_report

report = generate_medical_report(
    prediction="PNEUMONIA",
    confidence=98.54
)

print(report)