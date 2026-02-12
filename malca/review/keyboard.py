"""Keyboard shortcut handlers for Dash review app.

All tagging actions use a **leader-key prefix** so that single letters
stay available for future features:

    R <key>  →  toggle a reason tag  (multi-select)
    G <key>  →  set event class      (single-select; same key again clears)

Press Escape after the leader key to cancel.
"""

# ---------------------------------------------------------------------------
# Prefix-key maps.  The first keypress (leader) selects the group, the
# second keypress selects the item within that group.
# ---------------------------------------------------------------------------

# R prefix → reason tags (multi-select toggle)
REASON_PREFIX_KEY = 'r'
REASON_KEY_MAP = {
    'c': 'clean_event',
    'm': 'multi_camera_support',
    'i': 'interesting_morphology',
    'p': 'periodic_contaminant',
    'a': 'camera_artifact',
    'k': 'known_object_nearby',
    'f': 'needs_followup_data',
}

# G prefix → event class (single-select)
CLASS_PREFIX_KEY = 'g'
CLASS_KEY_MAP = {
    'd': 'circumstellar_dust',
    'l': 'microlensing',
    'f': 'flare',
    'e': 'eclipsing_binary',
    'i': 'instrumental',
    'u': 'unknown_interesting',
    'n': 'not_real',
}

# All prefix leader keys (used by app.py to detect prefix entry)
PREFIX_KEYS = {REASON_PREFIX_KEY, CLASS_PREFIX_KEY}

# ---------------------------------------------------------------------------
# Single-key shortcuts (no prefix required)
# ---------------------------------------------------------------------------
KEYBOARD_SHORTCUTS = {
    # Navigation
    'n': 'next_candidate',
    'N': 'next_candidate',
    'p': 'previous_candidate',
    'P': 'previous_candidate',
    'j': 'jump_to_index',
    'J': 'jump_to_index',

    # Scoring (instant save)
    '0': 'set_score_0',
    '1': 'set_score_1',
    '2': 'set_score_2',
    '3': 'set_score_3',
    '4': 'set_score_4',
    '5': 'set_score_5',

    # Prefix leaders are handled by the state machine in app.py

    # Followup flag toggle
    'f': 'toggle_followup',
    'F': 'toggle_followup',

    # Actions
    's': 'save_review',
    'S': 'save_review',
    'd': 'save_and_next',
    'D': 'save_and_next',

    # UI Control
    't': 'toggle_sidebar',
    'T': 'toggle_sidebar',
    'a': 'toggle_recent_activity',
    'A': 'toggle_recent_activity',
    '?': 'show_shortcuts',
}

HELP_TEXT = """
Navigation:
  N - Next | P - Previous | J - Jump

Scoring (instant save, clickable):
  0-5 - Set interest score

Event Class (G then key, single-select, clickable):
  G D - circumstellar dust   G L - microlensing
  G F - flare                G E - eclipsing binary
  G I - instrumental         G U - unknown interesting
  G N - not real             (Esc cancels)

Reason Tags (R then key, multi-select, clickable):
  R C - clean event          R M - multi camera support
  R I - interesting morph    R P - periodic contaminant
  R A - camera artifact      R K - known object nearby
  R F - needs followup data  (Esc cancels)

Status:
  F - Toggle needs-followup flag
  (status auto-set to reviewed on save)

Actions:
  S - Save | D - Done (Save + Next)
  M - Enter notes (Esc to exit)

UI:
  T - Toggle sidebar | A - Toggle activity
  Plot controls are in-GUI (preset/actions/native export)
  ? - Show this help
"""


def handle_key_action(key, current_idx, queue_size, conn, candidate_id):
    """
    Handle keyboard action and return new state.

    Returns:
        tuple: (new_idx, notification_msg, should_save)
    """
    action = KEYBOARD_SHORTCUTS.get(key)

    if not action:
        return current_idx, f"Unknown key: {key}", False

    # Navigation
    if action == 'next_candidate':
        new_idx = min(current_idx + 1, queue_size - 1)
        return new_idx, "→ Next", False

    elif action == 'previous_candidate':
        new_idx = max(0, current_idx - 1)
        return new_idx, "← Previous", False

    # Scoring (instant save)
    elif action.startswith('set_score_'):
        score = int(action[-1])
        return current_idx, f"✓ Score: {score}", True

    # Actions
    elif action == 'save_review':
        return current_idx, "✓ Saved", True

    elif action == 'save_and_next':
        new_idx = min(current_idx + 1, queue_size - 1)
        return new_idx, "✓ Saved + Next →", True

    # UI Control (handled by other callbacks, included for completeness)
    elif action == 'toggle_sidebar':
        return current_idx, "Sidebar toggled", False

    elif action == 'toggle_recent_activity':
        return current_idx, "Activity toggled", False

    elif action == 'show_shortcuts':
        return current_idx, "Help displayed", False

    # Reason toggles and followup are handled inline in app.py
    return current_idx, "", False
