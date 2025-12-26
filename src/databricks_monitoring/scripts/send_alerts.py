"""Send alerts based on monitoring thresholds."""

import argparse
import sys
from typing import List

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from databricks_monitoring.monitoring.config import load_config


def check_drift_threshold(
    spark: SparkSession,
    drift_table: str,
    threshold: float,
    days: int = 1,
) -> List[dict]:
    """Check if drift exceeds threshold.

    Args:
        spark: SparkSession
        drift_table: Drift metrics table name
        threshold: Drift threshold
        days: Number of days to check

    Returns:
        List of violations
    """
    try:
        drift_df = spark.table(drift_table)

        # Filter to recent data and high drift
        violations = drift_df.filter(
            (F.col("window_end") >= F.date_sub(F.current_date(), days))
            & (F.col("drift_score") > threshold)
        )

        violations_list = violations.select(
            "column_name",
            "drift_score",
            "drift_type",
            "window_end",
        ).collect()

        return [
            {
                "column": row["column_name"],
                "drift_score": row["drift_score"],
                "drift_type": row["drift_type"],
                "window_end": row["window_end"],
            }
            for row in violations_list
        ]

    except Exception as e:
        logger.warning(f"Could not check drift threshold: {e}")
        return []


def send_email_alert(
    recipients: List[str],
    subject: str,
    body: str,
) -> None:
    """Send email alert.

    Args:
        recipients: List of email addresses
        subject: Email subject
        body: Email body

    Note:
        This is a placeholder. Actual implementation requires
        SMTP configuration or Databricks email API.
    """
    logger.info(f"Sending email alert to: {', '.join(recipients)}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Body:\n{body}")

    # TODO: Implement actual email sending
    # Options:
    # 1. Use Databricks SQL Alerts API
    # 2. Use SMTP library (smtplib)
    # 3. Use SendGrid/AWS SES API
    # 4. Use Azure Communication Services

    logger.warning("Email sending not implemented - this is a placeholder")


def main():
    """Check monitoring metrics and send alerts if thresholds exceeded."""
    parser = argparse.ArgumentParser(description="Send monitoring alerts")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--environment", required=True, help="Environment (dev, aut, prod)")
    parser.add_argument("--monitoring-schema", default="monitoring", help="Monitoring schema name")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Monitoring Alerts Check")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(args.environment)

        # Initialize Spark
        spark = SparkSession.builder.getOrCreate()

        # Construct table names
        drift_table = f"{args.catalog}.{args.monitoring_schema}.drift_metrics"

        # Check each alert configuration
        alerts_triggered = []

        for alert_config in config.monitoring.alerts:
            logger.info(f"Checking alert: {alert_config.metric}")

            if alert_config.metric == "prediction_drift":
                # Check prediction drift
                violations = check_drift_threshold(
                    spark=spark,
                    drift_table=drift_table,
                    threshold=alert_config.threshold,
                    days=1,
                )

                if violations:
                    logger.warning(f"⚠ Alert triggered: {len(violations)} columns exceed drift threshold")

                    alerts_triggered.append({
                        "alert": alert_config,
                        "violations": violations,
                    })

                    # Prepare alert message
                    violation_details = "\n".join([
                        f"  - {v['column']}: {v['drift_score']:.4f} ({v['drift_type']})"
                        for v in violations
                    ])

                    subject = f"[{args.environment.upper()}] Prediction Drift Alert - {len(violations)} columns"
                    body = f"""
Drift Alert Triggered

Environment: {args.environment}
Catalog: {args.catalog}
Schema: {args.schema}
Threshold: {alert_config.threshold}

Violations:
{violation_details}

Please investigate the drift in these features and take appropriate action.

Dashboard: Check Lakehouse Monitoring dashboard for detailed analysis.
"""

                    # Send alert
                    send_email_alert(
                        recipients=alert_config.recipients,
                        subject=subject,
                        body=body,
                    )

                else:
                    logger.info("✓ No drift violations detected")

            elif alert_config.metric == "data_quality_score":
                # TODO: Implement data quality checks
                logger.info("Data quality checks not yet implemented")

            else:
                logger.warning(f"Unknown alert metric: {alert_config.metric}")

        # Summary
        logger.info("=" * 80)
        if alerts_triggered:
            logger.warning(f"Alerts Triggered: {len(alerts_triggered)}")
            for alert in alerts_triggered:
                logger.warning(f"  - {alert['alert'].metric}: {len(alert['violations'])} violations")
        else:
            logger.info("✓ No alerts triggered - all metrics within thresholds")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Alert check failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
