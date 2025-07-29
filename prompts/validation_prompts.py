# prompts/validation_prompts.py

RESUME_VALIDATION_PROMPT = """
<task>
Compare the original user data with the generated resume JSON to detect and correct hallucinations or inaccuracies. Ensure that the generated resume content aligns with the original user data and does not include fabricated or unsupported information.
</task>

<original_user_data>
{{ original_user_data }}
</original_user_data>

<generated_resume_json>
{{ generated_resume_json }}
</generated_resume_json>

<instructions>
1. Identify discrepancies between the original user data and the generated resume JSON.
2. Correct any hallucinations or inaccuracies in the generated resume JSON.
3. Ensure that all corrected content is supported by the original user data.
4. Return the validated and corrected resume JSON.
</instructions>

{{ format_instructions }}
"""