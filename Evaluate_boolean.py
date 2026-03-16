class Evaluate_Boolean:
    def get_alphabets_str(num):
        """
        This function takes an integer and returns a string of alphabets starting from 'a' up to the given number.
        """
        alphabets = ''
        for i in range(num):
            alphabets += chr(i+97)
        return alphabets

    def evaluate_boolean_function(expression, variable_values):
        """
        Evaluate a boolean function expressed as a sum of products expression, given the values of its variables.

        Parameters:
            expression (list of lists of str): A sum of products expression where each inner list represents a product
                of variables, and each variable is represented by a string. The inner list can also contain negations of
                variables, represented by a string with a tilde (~) before the variable name.
            variable_values (dict of str to bool): A dictionary where the keys are variable names (without the tilde)
                and the values are the boolean values of the variables.

        Returns:
            bool: The boolean value of the expression given the variable values.
        """
        result = False
        for product in expression:
            product_result = True
            for variable in product:
                negation = False
                if variable.startswith("~"):
                    negation = True
                    variable = variable[1:]
                if variable not in variable_values:
                    raise ValueError(f"Unknown variable: {variable}")
                variable_value = variable_values[variable]
                if negation:
                    variable_value = not variable_value
                product_result = product_result and variable_value
            result = result or product_result
        return result
