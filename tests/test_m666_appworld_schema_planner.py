from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.evaluate_appworld_checkpoint import _schema_ground_appworld_api_step


def _world() -> SimpleNamespace:
    def params(*names: str) -> list[dict[str, object]]:
        return [
            {"name": name, "required": name in {"artist_id", "access_token"}, "default": None}
            for name in names
        ]

    docs = {
        "spotify": {
            "show_current_song": {
                "description": "Show details of the current song on the queue.",
                "parameters": params("access_token"),
            },
            "show_artist": {
                "description": "Get details of a specific artist.",
                "parameters": params("artist_id"),
            },
            "search_artists": {
                "description": "Search for artists with a query.",
                "parameters": params("query", "min_follower_count"),
            },
            "search_songs": {
                "description": "Search for songs with a query.",
                "parameters": params("query", "artist_id", "min_play_count"),
            },
            "add_to_queue": {
                "description": "Add a song to the music player song queue.",
                "parameters": params("access_token", "song_id", "album_id", "playlist_id"),
            },
        }
    }
    return SimpleNamespace(task=SimpleNamespace(api_docs=docs))


def _step(prompt: str) -> str:
    return _schema_ground_appworld_api_step(
        model=None,
        tokenizer=None,
        world=_world(),
        prompt=prompt,
        allow_completion=True,
        lexical_first=True,
    ) or ""


def test_queue_planner_preserves_names_thresholds_and_single_target() -> None:
    instruction = "Add all the songs from Lily Moon that have been played over 980 times to my Spotify player queue."
    first = _step(instruction)
    assert first == "apis.spotify.search_artists(query='Lily Moon')"

    artist_result = json.dumps(
        {"value": [{"artist_id": 34, "name": "Lily Moon"}, {"artist_id": 2, "name": "Other"}]},
        sort_keys=True,
    )
    second = _step(
        f'{instruction}\nASSISTANT: [run_python({json.dumps({"app": "spotify", "api": "search_artists", "arguments": {"query": "Lily Moon"}}, sort_keys=True, separators=(",", ":"))})]\n'
        f"TOOL_RESULT: {artist_result}\nNext required action:"
    )
    assert second == "apis.spotify.search_songs(artist_id=34, min_play_count=980, query='Lily Moon')"

    songs_result = json.dumps(
        {"value": [{"album_id": 10, "song_id": 311, "title": "Infinite Dreams"}]},
        sort_keys=True,
    )
    third = _step(
        f'{instruction}\nASSISTANT: [run_python({json.dumps({"app": "spotify", "api": "search_songs", "arguments": {"artist_id": 34, "min_play_count": 980, "query": "Lily Moon"}}, sort_keys=True, separators=(",", ":"))})]\n'
        f"TOOL_RESULT: {songs_result}\nNext required action:"
    )
    assert third == "apis.spotify.add_to_queue(song_id=311)"


def test_follower_planner_uses_live_id_then_completion_answer() -> None:
    instruction = "How many people follow the artist of the currently playing song on Spotify?"
    assert _step(instruction) == "apis.spotify.show_current_song()"
    prompt = (
        f'{instruction}\nASSISTANT: [run_python({json.dumps({"app": "spotify", "api": "show_current_song", "arguments": {}}, sort_keys=True, separators=(",", ":"))})]\n'
        'TOOL_RESULT: {"value": {"artists": [{"id": 24, "name": "Liam Palmer"}]}}\n'
        "Next required action:"
    )
    assert _step(prompt) == "apis.spotify.show_artist(artist_id=24)"
    prompt += (
        '\nASSISTANT: [run_python({"api":"show_artist","app":"spotify","arguments":{"artist_id":24}})]\n'
        'TOOL_RESULT: {"value": {"follower_count": 20}}\nNext required action:'
    )
    assert _step(prompt) == "apis.supervisor.complete_task(answer=20, status='success')"
