"""
dashboard/auth.py
==================
Implements the role split described in Section 3.6.4: "Researchers get
access to the full data plus model inspection, while institutional staff
only see an aggregated cohort view and individual predictions, without
the raw data."

DECISION: this is a role *selector*, not real authentication. A genuine
login system (institutional SSO/OAuth, per-user credentials, audit
logging of who viewed what) is exactly the kind of thing Section 3.5's
ethical framework requires before any real student data reaches this
dashboard, and it is also inherently institution-specific (it has to plug
into whatever identity provider the deploying university already runs).
Building a real auth layer here would mean either (a) inventing a
throwaway login system that gives a false sense of security for a
research prototype that will never hold real student data, or (b) trying
to guess at a specific institution's SSO integration with nothing to
integrate against. Section 3.6.4 itself describes the mechanism as "a
role based control layer, merged with the Streamlit session state" --
which is exactly what this module does -- without specifying a particular
identity provider, which is consistent with treating the *real* auth
integration as a deployment-time concern for whichever institution adopts
the system (flagged again in HANDOVER.md).

What this module DOES enforce faithfully is the *visibility contract*:
once a role is selected, dashboard/app.py and the view modules gate what
each role can see exactly as Section 3.6.4 specifies, including which
demographic/raw fields are hidden from institutional staff.
"""
import streamlit as st

ROLE_RESEARCHER = "researcher"
ROLE_INSTITUTIONAL_STAFF = "institutional_staff"


def render_role_selector() -> str:
    st.sidebar.markdown("### Access role")
    role_label = st.sidebar.radio(
        "Viewing as",
        options=["Researcher (full access)", "Institutional Staff (restricted)"],
        index=1,
        help=(
            "Researchers can inspect raw EI/engagement scores, demographics, "
            "the SMOTE/baseline comparison, and the data-quality log. "
            "Institutional Staff see cohort-level predictions and individual "
            "risk flags only, per Section 3.5/3.6.4 of the proposal -- no "
            "raw survey scores or demographic breakdowns."
        ),
    )
    role = ROLE_RESEARCHER if role_label.startswith("Researcher") else ROLE_INSTITUTIONAL_STAFF
    st.session_state["role"] = role
    if role == ROLE_INSTITUTIONAL_STAFF:
        st.sidebar.caption(
            "Restricted view: predictions and intervention tracking only. "
            "Raw scores and demographic subgroup analysis are hidden."
        )
    else:
        st.sidebar.caption("Full access: raw data, model internals, and fairness reporting visible.")
    return role


def is_researcher() -> bool:
    return st.session_state.get("role") == ROLE_RESEARCHER
