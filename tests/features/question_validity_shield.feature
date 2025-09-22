@validation @question_validity
Feature: Question Validity Shield
  As a domain-specific AI assistant
  I want to validate that questions are related to OpenShift and Kubernetes
  So that I only respond to relevant technical queries

  Background:
    Given the llama-stack is running on "http://localhost:8321"
    And the question validity shield is configured

  Scenario Outline: Valid technical questions are allowed
    When I send the question "<question>" to the question validity shield
    Then the question should be allowed
    And no violation should be reported

    Examples:
      | question                                                    |
      | Can you help me deploy an application on OpenShift?        |
      | How do I configure OpenShift routes for my service?        |
      | What's the best way to configure OpenShift networking?     |
      | Can you show me how to create a Kubernetes service?        |

  Scenario Outline: Invalid non-technical questions are blocked
    When I send the question "<question>" to the question validity shield
    Then the question should be blocked
    And a violation should be reported
    And the response should contain the invalid question message

    Examples:
      | question                                |
      | What's the weather like today?          |
      | How do I make a chocolate cake?         |
      | What's the capital of France?           |
      | What's the best restaurant in town?     |

  Scenario: Custom invalid question response
    When I send an invalid question "What's your favorite color?"
    Then the question should be blocked
    And the response should contain the text "OpenShift Lightspeed assistant"
    And the response should contain the text "questions about OpenShift"
