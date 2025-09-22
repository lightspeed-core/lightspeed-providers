"""
Step definitions for question validity shield tests.
"""

from behave import given, when, then


@given("the question validity shield is configured")
def step_given_question_validity_shield_configured(context):
    """Verify the question validity shield is configured."""
    # This is implicitly verified by the service being up
    context.question_validity_shield_id = "lightspeed_question_validity-shield"


@when('I send the question "{question}" to the question validity shield')
def step_when_send_question_to_validity_shield(context, question):
    """Send a question to the question validity shield."""
    context.original_question = question
    context.client.run_shield(context.question_validity_shield_id, question)


@then("the question should be allowed")
def step_then_question_allowed(context):
    """Verify the question is allowed (no violation)."""
    assert (
        context.client.last_response.status_code == 200
    ), f"Expected 200, got {context.client.last_response.status_code}"

    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    assert (
        not violation
    ), f"Question should be allowed but violation was reported: {violation}"


@then("no violation should be reported")
def step_then_no_violation(context):
    """Verify no violation is reported."""
    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    assert not violation, f"No violation should be reported, but got: {violation}"


@then("the question should be blocked")
def step_then_question_blocked(context):
    """Verify the question is blocked (violation reported)."""
    assert (
        context.client.last_response.status_code == 200
    ), f"Expected 200, got {context.client.last_response.status_code}"

    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    assert violation, "Question should be blocked but no violation was reported"


@then("a violation should be reported")
def step_then_violation_reported(context):
    """Verify a violation is reported."""
    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    assert violation, "A violation should be reported but none was found"


@then("the response should contain the invalid question message")
def step_then_response_contains_invalid_message(context):
    """Verify the response contains the invalid question message."""
    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    assert violation, "Violation should be present"

    violation_message = violation.get("user_message") or violation.get("message", "")
    assert (
        "questions about OpenShift" in violation_message
    ), f"Expected invalid question message, got: '{violation_message}'"


@when('I send an invalid question "{question}"')
def step_when_send_invalid_question(context, question):
    """Send an invalid question to the shield."""
    context.original_question = question
    context.client.run_shield(context.question_validity_shield_id, question)


@then('the response should contain the text "{expected_text}"')
def step_then_response_contains_text(context, expected_text):
    """Verify the response contains specific text."""
    response_data = context.client.last_response_data
    violation = response_data.get("violation")

    if violation:
        # Handle both "message" and "user_message" keys
        violation_message = violation.get("user_message") or violation.get(
            "message", ""
        )
        assert (
            expected_text in violation_message
        ), f"Expected '{expected_text}' in violation message: '{violation_message}'"
    else:
        # Check in other parts of the response if no violation
        response_str = str(response_data)
        assert (
            expected_text in response_str
        ), f"Expected '{expected_text}' in response: '{response_str}'"
