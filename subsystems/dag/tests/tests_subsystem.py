from subsystems.dag import DataAggregationSubsystem
from subsystems.dag.core.registry import PluginRegistry


def test_subsystem_exposes_full_name():
    assert DataAggregationSubsystem.name == 'Data Aggregation'


def test_plugin_registry_creates_registered_plugin():
    registry = PluginRegistry()
    registry.register('example', dict)

    assert registry.create('example') == {}
    assert registry.names() == ['example']
