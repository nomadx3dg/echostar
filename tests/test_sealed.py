import sealed


def test_roundtrip_and_wrong_code():
    blob = sealed.seal(b"model bytes" * 100, "Some-Code 12")
    assert b"model bytes" not in blob
    assert sealed.unseal(blob, "somecode12") == b"model bytes" * 100      # case and punctuation do not matter
    assert sealed.unseal(blob, "somecode13") is None
    assert sealed.unseal(blob[:-1] + bytes([blob[-1] ^ 1]), "somecode12") is None   # damaged
    assert sealed.unseal(b"junk", "somecode12") is None


def test_unseal_suffixes_ignores_noise_before_the_code():
    blob = sealed.seal(b"x", "abcd1234")
    assert sealed.unseal_suffixes(blob, "zzzzabcd1234") == b"x"
    assert sealed.unseal_suffixes(blob, "abcd1234zz") is None


class FakeBrain:
    pass


def test_sealed_bot_becomes_available_only_after_the_unlock_word(tmp_path):
    from test_ui_server import CODE, make_app
    app = make_app(tmp_path, fischer_available=False)
    calls = []

    def unseal(typed):
        calls.append(typed)
        return FakeBrain()
    app.bots["fischer"].unseal = unseal
    app.unlock("nothing")
    assert calls == [] and not app.bots["fischer"].available          # a wrong word never touches the sealed file
    app.unlock(CODE)
    assert calls == [CODE] and app.bots["fischer"].available
    app.unlock(CODE)
    assert calls == [CODE]                                            # loaded once
    app.new_game("fischer", "white", "sharp")
