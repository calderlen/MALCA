"""Keyboard shortcut handlers for Dash review app.

Single-key class shortcuts — letter keys set the event class directly
(pressing the same key again clears to unclassified).  Non-letter keys
handle navigation, saving, and UI control.
"""

# ---------------------------------------------------------------------------
# Class key map — single press sets the class (toggle: same key clears)
# ---------------------------------------------------------------------------
CLASS_KEY_MAP = {
    'd': 'dipper',
    'm': 'microlensing',
    'f': 'flare',
    'l': 'ltv',
    'u': 'unknown_interesting',
    'i': 'instrumental',
    'o': 'other',
}

# ---------------------------------------------------------------------------
# Single-key shortcuts (non-letter keys for navigation / actions)
# ---------------------------------------------------------------------------
KEYBOARD_SHORTCUTS = {
    # Scoring (instant save)
    '1': 'set_score_1',
    '2': 'set_score_2',
    '3': 'set_score_3',
    '4': 'set_score_4',

    # Navigation / actions
    'Backspace': 'previous_candidate',
    'Enter': 'save_and_next',
    'Tab': 'next_candidate',
    '.': 'save_review',
    ',': 'toggle_followup',

    # UI Control
    'Escape': 'toggle_sidebar',
    '?': 'show_shortcuts',
}

HELP_TEXT = """
Classes (single key, toggle):
  [D] dipper         [M] microlensing
  [F] flare          [L] ltv
  [U] unknown interesting
  [I] instrumental   [O] other

Confidence (instant save):
  [1]-[4] Set confidence level

Navigation:
  [Backspace] Previous  [Tab] Next (no save)
  [Enter] Save + Next

Actions:
  [.] Save  [,] Toggle followup

UI:
  [Esc] Toggle sidebar  [Shift+R] Refresh queue  [?] Show this help
"""


def handle_key_action(key, current_idx, queue_size, conn, candidate_id):
    """
    Handle keyboard action and return new state.

    Returns:
        tuple: (new_idx, notification_msg, should_save)
    """
    action = KEYBOARD_SHORTCUTS.get(key)

    if not action:
        return current_idx, "", False

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
        return current_idx, f"✓ Confidence: {score}", True

    # Actions
    elif action == 'save_review':
        return current_idx, "✓ Saved", True

    elif action == 'save_and_next':
        new_idx = min(current_idx + 1, queue_size - 1)
        return new_idx, "✓ Saved + Next →", True

    # UI Control (handled by other callbacks, included for completeness)
    elif action == 'toggle_sidebar':
        return current_idx, "Sidebar toggled", False

    elif action == 'show_shortcuts':
        return current_idx, "Help displayed", False

    # Class toggles and followup are handled inline in app.py
    return current_idx, "", False
