from backend.database.database import get_connection


def save_report(
    filename,
    prediction,
    confidence,
    report
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO reports
        (
            filename,
            prediction,
            confidence,
            report
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            filename,
            prediction,
            confidence,
            report,
        ),
    )

    conn.commit()
    conn.close()

    print("Report saved successfully.")


def get_all_reports():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            prediction,
            confidence,
            created_at
        FROM reports
        ORDER BY id DESC
        """
    )

    reports = cursor.fetchall()

    conn.close()

    return reports


def get_report(report_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM reports
        WHERE id=?
        """,
        (report_id,),
    )

    report = cursor.fetchone()

    conn.close()

    return report