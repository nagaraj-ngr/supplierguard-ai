from supplierguard.config import load_settings


def test_defaults_need_no_secrets():
    s = load_settings({})
    assert s.openai_api_key is None and not s.has_openai_key
    assert s.seed == 42 and s.embedder_kind == "auto"
    assert s.index_dir == s.data_dir / "index"


def test_reads_environment():
    s = load_settings({"OPENAI_API_KEY": " sk-test ", "SUPPLIERGUARD_SEED": "7",
                       "SUPPLIERGUARD_EMBEDDER": "HASH"})
    assert s.openai_api_key == "sk-test" and s.has_openai_key
    assert s.seed == 7 and s.embedder_kind == "hash"


def test_blank_key_counts_as_missing():
    assert not load_settings({"OPENAI_API_KEY": "   "}).has_openai_key


def test_invalid_embedder_rejected():
    try:
        load_settings({"SUPPLIERGUARD_EMBEDDER": "magic"})
    except ValueError as e:
        assert "SUPPLIERGUARD_EMBEDDER" in str(e)
    else:
        raise AssertionError("expected ValueError")
