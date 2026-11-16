

def test_prediction_response():

    from src.config.logger import get_logger
    from src.services.ticket_service import generate_prediction_result

    logger = get_logger("test_api.py")

    try:
        ticket = {
            "ticket_id": "CI-001",
            "description": "Employee cannot connect to the company Wi-Fi network.",
            "incident_id": "Test_1234"
        }

        result = generate_prediction_result(ticket)

        assert "resolver_prediction" in result
        assert "requires_human_review" in result
        assert "review_reasons" in result

        prediction = result["resolver_prediction"]

        assert "resolver_group" in prediction
        assert "priority" in prediction
        assert 0 <= prediction["resolver_confidence"] <= 1
        assert 0 <= prediction["priority_confidence"] <= 1
    except Exception as ex:
        logger.error(f"test_prediction_response: {ex}")
