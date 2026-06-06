# This file was mechanically split from malca.review.app; preserve behavior when editing.
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="MALCA Review",
    background_callback_manager=_background_callback_manager,
)


@app.server.after_request
def _disable_dash_runtime_cache(response):
    """Keep browser tabs from reusing stale Dash callback maps after code changes."""
    if request.path in {"/_dash-layout", "/_dash-dependencies", "/_dash-update-component"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
