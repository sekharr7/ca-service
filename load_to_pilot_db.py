#!/usr/bin/env python3
"""
Sync Script: Promote processed Current Affairs articles from dev (mntor_dev)
to test/pilot database (pilot_db).

Ensures all 4 tables:
- mtr_current_affairs_edk
- mtr_ca_highlights_edk
- mtr_ca_images_edk
- mtr_ca_kword_edk
are migrated while strictly maintaining original IDs.

Usage:
    ./.venv/bin/python load_to_pilot_db.py [--min-id 10108]
"""

import sys
from sqlalchemy import text
from current_affairs.database import get_db_session

TABLES = [
    "mtr_current_affairs_edk",
    "mtr_ca_highlights_edk",
    "mtr_ca_images_edk",
    "mtr_ca_kword_edk",
]

def verify_dev_records(db, min_id: int):
    """Ensure no corrupted, error, or dummy records exist in mntor_dev before promoting."""
    check_query = text("""
        SELECT id, topic, content, source
        FROM mntor_dev.mtr_current_affairs_edk
        WHERE id > :min_id
          AND (
            topic REGEXP 'HTTP|403 Forbidden|URL Not|Event Not|cannot be accessed|not identifiable|\\.pdf|\\.png|\\.jpg|Dummy|Placeholder'
            OR content IS NULL
            OR LENGTH(content) < 20
            OR topic IS NULL
            OR LENGTH(topic) < 10
          )
    """)
    bad_rows = db.execute(check_query, {"min_id": min_id}).fetchall()
    if bad_rows:
        bad_ids = [str(r[0]) for r in bad_rows]
        print(f"\n[BLOCKED] Promotion aborted: Found {len(bad_rows)} corrupted/dummy record(s) in mntor_dev (IDs: {', '.join(bad_ids)}):", file=sys.stderr)
        for r in bad_rows:
            print(f"   - ID {r[0]}: Topic='{r[1]}', Source='{r[3]}'", file=sys.stderr)
        raise ValueError("Cannot promote corrupted records to pilot_db. Clean dev database first.")


def promote_dev_to_pilot(min_id: int = 10108):
    db = get_db_session()
    print(f"[SYNC] Starting promotion from mntor_dev to pilot_db for IDs > {min_id}...\n")

    total_synced = {}

    try:
        # Pre-migration quality audit
        verify_dev_records(db, min_id)
        print("  ✓ Pre-migration quality audit passed: 0 corrupted or dummy records in mntor_dev.\n")

        for table in TABLES:
            query = f"""
            INSERT INTO pilot_db.{table}
            SELECT *
            FROM mntor_dev.{table} src
            WHERE NOT EXISTS (
              SELECT 1
              FROM pilot_db.{table} tgt
              WHERE tgt.id = src.id
            ) AND src.id > :min_id
            """
            result = db.execute(text(query), {"min_id": min_id})
            inserted_count = result.rowcount
            total_synced[table] = inserted_count
            print(f"  ✓ {table:<25}: {inserted_count} new record(s) inserted")

        db.commit()
        print("\n[SYNC] All tables committed successfully to pilot_db.")

        # Summary check
        print("\n[VERIFICATION] Current record status in pilot_db:")
        for table in TABLES:
            count = db.execute(text(f"SELECT COUNT(*) FROM pilot_db.{table} WHERE id > :min_id"), {"min_id": min_id}).scalar()
            max_id = db.execute(text(f"SELECT MAX(id) FROM pilot_db.{table}")).scalar()
            print(f"  - pilot_db.{table:<25}: Total > {min_id} = {count} (Max ID: {max_id})")

    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Sync failed: {e}", file=sys.stderr)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    min_id = 10108
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        min_id = int(sys.argv[1])
    promote_dev_to_pilot(min_id=min_id)
