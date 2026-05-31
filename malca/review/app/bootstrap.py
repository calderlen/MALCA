# This file was mechanically split from malca.review.app; preserve behavior when editing.
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="MALCA Review",
    background_callback_manager=_background_callback_manager,
)

