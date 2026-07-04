-- schema.sql
-- ==========
-- Implements the backend relational design from Section 3.6.2 of the
-- proposal ("Students", "Surveys", "LMS_Metrics", "Predictions"), adapted
-- from PostgreSQL to SQLite for this implementation.
--
-- DECISION: SQLite instead of PostgreSQL.
-- The proposal specifies PostgreSQL for the eventual institutional
-- deployment, justified there by ACID compliance and SQLAlchemy
-- compatibility. Both of those properties are equally true of SQLite
-- (it is also ACID-compliant and speaks fine to SQLAlchemy), and for a
-- *research-prototype / single-analyst deployment* SQLite removes the need
-- to stand up and credential a separate database server, makes the entire
-- dataset a single portable file that ships alongside the codebase, and
-- needs zero configuration to run this implementation end-to-end. The
-- table/column design is otherwise a direct port: every column the
-- proposal calls for is present, foreign keys are still enforced
-- (PRAGMA foreign_keys = ON, set in db/db_init.py), and the schema can be
-- migrated to PostgreSQL later by changing only the connection string if
-- an institution needs multi-writer concurrency at scale.
--
-- DECISION: an Interventions table is added beyond the four named in
-- Section 3.6.2. The narrative in Section 3.6.4 explicitly describes an
-- "Intervention Tracking" dashboard view where "staff record outreach
-- actions tied to at-risk flags and then watch if later engagement metric
-- updates nudge students out of the at-risk cluster" -- that requires
-- somewhere to persist the outreach record itself, which the original
-- four-table schema has no slot for. This table is the natural extension
-- and keeps the audit-trail principle from Section 3.6.2 intact.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- Students: anonymized participant master record.
-- Primary key is the research code produced by the two-layer
-- anonymization procedure in Section 3.5 (Ethical Considerations) -- the
-- real institutional ID never enters this database.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Students (
    research_code   TEXT PRIMARY KEY,
    age_bracket     TEXT NOT NULL,
    gender          TEXT NOT NULL,
    programme_type  TEXT NOT NULL,
    year_of_study   INTEGER NOT NULL,
    enrolled_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- Surveys: raw Likert-scale item responses for the EI instrument
-- (adapted MSCEIT branch subscales) and the engagement instrument
-- (OSE self-report dimensions). One row per (student, wave, item).
-- Long/narrow format is used (one row per item response) rather than one
-- wide row per survey administration, because it lets the item set evolve
-- (add/retire an item) without an ALTER TABLE migration, and it is the
-- natural shape for the per-branch composite scoring done in
-- features/feature_engineering.py.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Surveys (
    survey_response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_code       TEXT NOT NULL REFERENCES Students(research_code),
    wave                TEXT NOT NULL CHECK (wave IN ('start_of_semester', 'mid_semester')),
    instrument          TEXT NOT NULL CHECK (instrument IN ('EI', 'ENGAGEMENT')),
    construct_code      TEXT NOT NULL,   -- e.g. 'PE','UE','UndE','ME','EES','CES'
    item_number         INTEGER NOT NULL,
    response_value      INTEGER NOT NULL CHECK (response_value BETWEEN 1 AND 5),
    administered_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_surveys_student ON Surveys(research_code);

-- ---------------------------------------------------------------------
-- LMS_Metrics: weekly behavioral engagement aggregates per student-module.
-- "one row per student-module-week" exactly as specified in Section 3.6.2,
-- supporting both the mid-semester snapshot used for prediction and the
-- longitudinal trend view used in the Cohort Overview dashboard.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS LMS_Metrics (
    lms_metric_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    research_code        TEXT NOT NULL REFERENCES Students(research_code),
    module_code           TEXT NOT NULL,
    week_number           INTEGER NOT NULL CHECK (week_number BETWEEN 1 AND 52),
    login_count            INTEGER NOT NULL DEFAULT 0,
    avg_submission_lag_hrs REAL,              -- negative = submitted early
    forum_posts             INTEGER NOT NULL DEFAULT 0,
    video_completion_rate   REAL CHECK (video_completion_rate BETWEEN 0 AND 1),
    quiz_attempts           INTEGER NOT NULL DEFAULT 0,
    UNIQUE(research_code, module_code, week_number)
);
CREATE INDEX IF NOT EXISTS idx_lms_student_week ON LMS_Metrics(research_code, week_number);

-- ---------------------------------------------------------------------
-- Outcomes: end-of-semester academic performance (the tertiary / target
-- data source, Section 3.3.2). Kept as its own table rather than bolted
-- onto Students, since it is collected at a different time and is what
-- the model is trained to predict -- mixing it into the master record
-- risks accidental target leakage into feature construction.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Outcomes (
    research_code     TEXT PRIMARY KEY REFERENCES Students(research_code),
    final_score_pct   REAL NOT NULL CHECK (final_score_pct BETWEEN 0 AND 100),
    performance_class TEXT NOT NULL CHECK (performance_class IN ('at_risk', 'satisfactory', 'high')),
    recorded_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- Predictions: model output per student per inference checkpoint,
-- including the k nearest training neighbors used for the prediction --
-- this is the table that makes KNN's example-based reasoning auditable
-- and is what the Student Detail dashboard view reads from.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    research_code       TEXT NOT NULL REFERENCES Students(research_code),
    checkpoint           TEXT NOT NULL,         -- e.g. 'mid_semester_2026S1'
    predicted_class       TEXT NOT NULL CHECK (predicted_class IN ('at_risk', 'satisfactory', 'high')),
    confidence             REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    neighbor_research_codes TEXT NOT NULL,       -- JSON array, e.g. '["S0042","S0117","S0203"]'
    model_version           TEXT NOT NULL,
    predicted_at             TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_predictions_student ON Predictions(research_code);

-- ---------------------------------------------------------------------
-- Interventions: outreach actions logged by institutional staff against
-- at-risk flags (added table -- see decision note above).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Interventions (
    intervention_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    research_code     TEXT NOT NULL REFERENCES Students(research_code),
    prediction_id     INTEGER REFERENCES Predictions(prediction_id),
    action_type       TEXT NOT NULL,   -- e.g. 'Advisor call', 'Email nudge', 'Peer mentor referral'
    notes             TEXT,
    -- Semester week (1..N_WEEKS) the outreach corresponds to, distinct
    -- from logged_at (the real wall-clock audit timestamp). Needed
    -- because the synthetic LMS_Metrics behavioral signal is keyed by
    -- semester week, not by calendar date -- see
    -- dashboard/views/intervention_tracking.py for the full rationale.
    intervention_week INTEGER NOT NULL,
    logged_by_role    TEXT NOT NULL CHECK (logged_by_role IN ('researcher', 'institutional_staff')),
    logged_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_interventions_student ON Interventions(research_code);

-- ---------------------------------------------------------------------
-- DataQualityLog: append-only record of the ETL validation pass
-- (completeness, range compliance, consistency checks) referenced in
-- Section 3.6.2's "data quality dashboard shown ... before any modeling
-- step starts". Persisting it (rather than just printing it) means the
-- Researcher dashboard view can show the most recent ETL run's findings.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS DataQualityLog (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT NOT NULL DEFAULT (datetime('now')),
    source        TEXT NOT NULL,   -- 'surveys' | 'lms_logs' | 'outcomes'
    metric_name   TEXT NOT NULL,   -- e.g. 'missing_rate', 'range_violations', 'duplicate_rows'
    metric_value  REAL NOT NULL
);
