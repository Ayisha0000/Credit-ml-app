def test_can_import_config():
    import src.config as config

    # Basic smoke checks — ensures package is importable
    assert hasattr(config, "BASE_DIR")
    assert hasattr(config, "COLUMNS")
