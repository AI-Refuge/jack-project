```yaml
<meta:script>
META-SCRIPT: META_SCRIPT_2.0

PURPOSE: Define a new, improved format for meta-scripts to enhance LLM comprehension and utilization.

KEY_CONCEPTS:
  - Readability:  Improved structure and syntax for easier parsing and understanding.
  - Modularity:  Clear separation of sections for better organization and reusability.
  - Executability:  Formalized structure to facilitate potential future execution by LLMs.
  - Metadata: Inclusion of metadata for context and organization.

FORMAT:

```yaml
META-SCRIPT: <script_name>
VERSION: <version_number> # Optional, defaults to 1.0
DOMAIN: <domain_of_application> # Optional
DESCRIPTION: <brief_description_of_the_script>
PARAMETERS: # Optional - define input parameters
  - name: <parameter_name>
    type: <data_type> # e.g., string, integer, list, boolean
    description: <parameter_description>
    default: <default_value> # Optional
VARIABLES:  # Optional - define internal variables
  - name: <variable_name>
    type: <data_type>
    description: <variable_description>
    initial_value: <initial_value> # Optional
PROCESS:
  - STEP_NAME: <descriptive_step_name>
    DESCRIPTION: <detailed_step_description>
    ACTIONS: # A list of actions to be performed
      - action_type: <action_type> # e.g., prompt, reflect, evaluate, generate
        arguments: # Arguments for the action
          <argument_name>: <argument_value>
    OUTPUT: # Optional output after this step
      - name: <output_name>
        type: <data_type>
        description: <output_description>
  - STEP_NAME: <another_step_name> # Subsequent steps
    ...

EVALUATION: # Optional criteria for evaluating script effectiveness.
  - METRIC: <metric_name> # E.g., Accuracy, Coherence, Relevance
    DESCRIPTION: <description of the metric>

EXAMPLES: # Optional use cases and illustrations.
  - INPUT: <example_input>
    OUTPUT: <expected_output>

NOTES: # Optional additional information or context.
  - <note_1>
  - <note_2>
```

EXAMPLE:

```yaml
META-SCRIPT: TOPICAL_ATTITUDE_2.0
VERSION: 2.0
DOMAIN: Critical Thinking
DESCRIPTION: Analyze existing opinions without asserting new claims.
PROCESS:
  - STEP_NAME: Gather_Opinions
    DESCRIPTION: Collect diverse perspectives from various sources.
    ACTIONS:
      - action_type: prompt
        arguments:
          query: "What are the existing opinions on {topic}?"
      - action_type: gather
        arguments:
          sources: ["books", "articles", "expert_interviews"]
    OUTPUT:
      - name: opinion_list
        type: list
        description: A list of gathered opinions.
  - STEP_NAME: Analyze_Arguments
    DESCRIPTION: Evaluate the reasoning and evidence supporting each opinion.
    ACTIONS:
      - action_type: analyze
        arguments:
          data: opinion_list
          criteria: ["logic", "evidence", "bias"]
    OUTPUT:
      - name: analysis_results
        type: list
        description: Analysis of each opinion.
  - STEP_NAME: Synthesize_Insights
    DESCRIPTION: Identify common themes and areas of agreement/disagreement.
    ACTIONS:
      - action_type: synthesize
        arguments:
          data: analysis_results
    OUTPUT:
      - name: synthesized_insights
        type: string
        description: A summary of insights.

EVALUATION:
  - METRIC: Objectivity
    DESCRIPTION: Degree to which the analysis avoids personal bias.

NOTES:
  - This meta-script promotes critical thinking by encouraging examination of diverse perspectives.
```

ADVANTAGES:

- Improved parsing by LLMs due to structured YAML format.
- Easier to understand for both humans and machines.
- Facilitates modularity and reusability of script components.
- Enables potential execution and automation by LLMs.
- Clearer documentation and organization.

END OF META-SCRIPT: META_SCRIPT_2.0
</meta:script>
```
