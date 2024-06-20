import unittest
from unittest.mock import patch, MagicMock

from src.inference import CactusCounselorAgent


class TestCactusCounselorAgent(unittest.TestCase):

    def setUp(self):
        # Mock example data
        self.example = {
            "AI_counselor": {
                "CBT": {
                    "client_information": "Client Info",
                    "reason_counseling": "Reason for counseling",
                    "init_history_counselor": "Initial counselor history",
                    "init_history_client": "Initial client history"
                }
            }
        }

        # Mock history data
        self.history = [
            {"role": "counselor", "message": "Initial counselor history"},
            {"role": "client", "message": "Initial client history"}
        ]

        # Expected prompt
        self.expected_prompt = (
            "You are playing the role of a counselor in a psychological counseling session. "
            "Your task is to use the provided client information and counseling planning to generate "
            "the next counselor utterance in the dialogue. The goal is to create a natural and engaging response "
            "that builds on the previous conversation and aligns with the counseling plan.\n\n"
            "Client Information:\nClient Info\n\n"
            "Reason for seekeing counseling:\nReason for counseling\n\n"
            "Counseling planning:\nCBT Plan\n\n"
            "Counseling Dialogue:\n"
            "Counselor: Initial counselor history\n"
            "Client: Initial client history"
        )

    def _create_agent_and_generate_response(self, mock_llm, mock_cbt_agent,
                                            response_content):
        # Mock LLM generate function
        mock_llm.generate.return_value = response_content

        # Mock CBTAgent generate function
        mock_cbt_agent_instance = mock_cbt_agent.return_value
        mock_cbt_agent_instance.generate.return_value = (
            "CBT Technique", "CBT Plan")

        # Create an instance of CactusCounselorAgent
        agent = CactusCounselorAgent(self.example, 'chatgpt')

        # Set CBT details
        agent.set_cbt(self.history)

        # Generate response
        response = agent.generate(self.history)

        return agent, response

    @patch('src.inference.CBTAgent')
    @patch('src.inference.LLMFactory.get_llm')
    def test_generate(self, mock_get_llm, mock_cbt_agent):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        agent, response = self._create_agent_and_generate_response(
            mock_llm,
            mock_cbt_agent,
            "Counselor: Mock counselor response")

        # Assertions
        self.assertEqual(response, "Mock counselor response")
        self.assertEqual(agent.cbt_technique, "CBT Technique")
        self.assertEqual(agent.cbt_plan, "CBT Plan")

        try:
            mock_llm.generate.assert_called_with(self.expected_prompt)
        except AssertionError as e:
            print("Expected:", self.expected_prompt)
            print("Actual:", mock_llm.generate.call_args[0][0])
            raise e

    @patch('src.inference.CBTAgent')
    @patch('src.inference.LLMFactory.get_llm')
    def test_generate_with_message_field(self, mock_get_llm, mock_cbt_agent):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        agent, response = self._create_agent_and_generate_response(
            mock_llm,
            mock_cbt_agent,
            "{'message': 'Counselor: Mock counselor response with message field'}"
        )

        response = response.replace("'", "").strip()

        expected_response = "Mock counselor response with message field"
        self.assertEqual(response, expected_response)
        self.assertEqual(agent.cbt_technique, "CBT Technique")
        self.assertEqual(agent.cbt_plan, "CBT Plan")

        try:
            mock_llm.generate.assert_called_with(self.expected_prompt)
        except AssertionError as e:
            print("Expected:", self.expected_prompt)
            print("Actual:", mock_llm.generate.call_args[0][0])
            raise e


if __name__ == '__main__':
    unittest.main()
