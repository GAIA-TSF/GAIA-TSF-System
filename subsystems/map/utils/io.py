"""
Monitoring module (mock)

Represents:
- CUSUM
- Bayesian change point detection
"""


def run_monitoring(residuals, config):
    """
    Convert residuals → risk signals
    """
    print(f'[Monitoring] Running on residuals={residuals}')
    print('  - CUSUM')
    print('  - Bayesian CPD')

    return {'status': 'ok'}
