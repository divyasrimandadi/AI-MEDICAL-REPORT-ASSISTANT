from backend.services.database_service import (
    save_report,
    get_all_reports
)

save_report(
    filename="person1_virus_6.jpeg",
    prediction="PNEUMONIA",
    confidence=98.54,
    report="This is a test AI report."
)

reports = get_all_reports()

for report in reports:
    print(report)