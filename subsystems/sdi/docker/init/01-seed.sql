CREATE DATABASE geodata;

\connect geodata;

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS test;

CREATE TABLE IF NOT EXISTS test.fakesite_phmeasurements_2025_cybele (
                                                                        id SERIAL PRIMARY KEY,
                                                                        location geometry(Point, 4326) NOT NULL,
    ph NUMERIC(3,2) NOT NULL CHECK (ph BETWEEN 0 AND 14),
    measurement_date DATE NOT NULL
    );

INSERT INTO test.fakesite_phmeasurements_2025_cybele
(location, ph, measurement_date)
VALUES
    (ST_SetSRID(ST_MakePoint(14.5000, 50.1000), 4326), 2.15, '2025-10-01'),
    (ST_SetSRID(ST_MakePoint(14.5050, 50.1020), 4326), 3.20, '2025-10-03'),
    (ST_SetSRID(ST_MakePoint(14.4980, 50.0980), 4326), 2.85, '2025-10-05'),
    (ST_SetSRID(ST_MakePoint(14.5030, 50.1010), 4326), 3.75, '2025-10-07'),
    (ST_SetSRID(ST_MakePoint(14.4970, 50.1040), 4326), 2.45, '2025-10-09'),
    (ST_SetSRID(ST_MakePoint(14.5065, 50.0990), 4326), 3.10, '2025-10-11'),
    (ST_SetSRID(ST_MakePoint(14.5015, 50.0975), 4326), 2.95, '2025-10-13'),
    (ST_SetSRID(ST_MakePoint(14.5040, 50.1035), 4326), 3.55, '2025-10-15'),
    (ST_SetSRID(ST_MakePoint(14.4995, 50.1050), 4326), 2.60, '2025-10-17'),
    (ST_SetSRID(ST_MakePoint(14.5070, 50.1005), 4326), 3.35, '2025-10-19');
