
class SimpleKnowledgeEngine:
    """A simple, self-contained forward-chaining rule engine."""

    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.diagnosis = "No diagnosis could be made based on the provided symptoms."

    def reset(self):
        self.facts = set()
        self.diagnosis = "No diagnosis could be made based on the provided symptoms."

    def declare(self, fact):
        self.facts.add(fact)

    def run(self):
        new_facts_found = True
        while new_facts_found:
            new_facts_found = False
            for rule in self.rules:
                if rule['conditions'].issubset(self.facts):
                    if rule['conclusion'] not in self.facts:
                        self.facts.add(rule['conclusion'])
                        self.diagnosis = rule['conclusion']
                        new_facts_found = True


# Define all the rules for the expert system
# This is easily expandable
rules = [
    {
        'conditions': {'symptom_joint_pain', 'pain_worsens_with_activity_yes', 'age_over_50'},
        'conclusion': 'Osteoarthritis'
    },
    {
        'conditions': {'symptom_morning_stiffness_gt_30_mins', 'multiple_joints_affected_yes', 'symmetrical_joint_involvement_yes'},
        'conclusion': 'Rheumatoid Arthritis'
    },
    {
        'conditions': {'symptom_severe_pain_in_big_toe', 'joint_is_red_and_hot_yes', 'history_of_high_uric_acid_yes'},
        'conclusion': 'Gout'
    },
    # --- ADD YOUR 100+ RULES HERE IN THE SAME FORMAT ---
]
