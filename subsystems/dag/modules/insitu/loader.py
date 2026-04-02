from datetime import datetime


class InSituLoader:
    def load_csv(self, path: str):
        print(f'[InSituLoader] Loading CSV: {path}')

        timestamps = []
        values = []

        with open(path, 'r') as f:
            lines = f.readlines()[1:]  # skip header

            for line in lines:
                parts = line.strip().split(';')

                ts = datetime.strptime(parts[0], '%d.%m.%Y %H:%M:%S.%f')
                value = float(parts[4])

                timestamps.append(ts)
                values.append(value)

        return {
            'timestamps': timestamps,
            'values': values,
            'x': float(parts[2]),
            'y': float(parts[3]),
            'point_id': parts[1],
        }
