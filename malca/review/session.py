"""Session state management for Dash review app."""
from pathlib import Path
import hashlib

import pandas as pd

from malca.review.store import query_queue





class SessionState:
    """Manages cached queue and navigation state."""

    def __init__(self):
        self.queue_df = None
        self.queue_hash = None
        self.candidate_ids = []
        self.payloads = {}
        self.current_idx = 0

    def needs_refresh(self, filter_dict):
        """Check if queue needs to be refreshed based on filter changes."""
        new_hash = self._compute_hash(filter_dict)
        if new_hash != self.queue_hash:
            self.queue_hash = new_hash
            return True
        return False

    def _compute_hash(self, filter_dict):
        """Compute hash of filter parameters."""
        # Convert dict to sorted tuple for consistent hashing
        items = sorted(filter_dict.items())
        hash_str = str(items)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def load_queue(self, df, payloads):
        """Load queue DataFrame and payloads into cache."""
        self.queue_df = df
        self.candidate_ids = df['candidate_id'].tolist() if not df.empty else []
        self.payloads = payloads
        self.current_idx = 0

    def get_candidate(self, idx):
        """Get candidate at index (O(1) lookup)."""
        if not self.candidate_ids or idx < 0 or idx >= len(self.candidate_ids):
            return None, None

        candidate_id = self.candidate_ids[idx]
        payload = self.payloads.get(candidate_id, {})
        return candidate_id, payload

    def get_queue_size(self):
        """Get total queue size."""
        return len(self.candidate_ids)

    def get_progress_text(self):
        """Get progress text for display."""
        if not self.candidate_ids:
            return "[0/0]"
        return f"[{self.current_idx + 1}/{len(self.candidate_ids)}]"


def create_queue_data_dict(conn, filter_params):
    """
    Query queue and create data dict for Dash Store.

    Args:
        conn: Database connection
        filter_params: Dict of filter parameters

    Returns:
        dict: {
            'candidate_ids': list,
            'queue_size': int,
            'filter_hash': str
        }
    """


    # Query queue with filters (pass the whole dict through)
    df = query_queue(conn, filters=filter_params)

    # Extract candidate IDs
    candidate_ids = df['candidate_id'].tolist() if not df.empty else []

    # Compute filter hash
    filter_hash = hashlib.md5(str(sorted(filter_params.items())).encode()).hexdigest()

    return {
        'candidate_ids': candidate_ids,
        'queue_size': len(candidate_ids),
        'filter_hash': filter_hash
    }
