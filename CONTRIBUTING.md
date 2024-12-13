# Contributing to Spey

We welcome contributions to ``spey`` via
  [Pull Requests to our GitHub repository](https://github.com/SpeysideHEP/spey/pulls).
To get started, fork the ``main`` repository.

For anything beyond minimal fixes that do not require discussion, please first [open an issue](https://github.com/SpeysideHEP/spey/issues/new/choose)
to discuss your request with the development team.

If your proposed changes are more extensive than a few lines of code, please create a draft pull request. This draft should include the context of the change, a description, and the benefits of the implementation.

- For changes within the Python interface of the program, please run standard tests and write additional tests if necessary.
- Ensure you add examples demonstrating the new implementation.
- Specify any drawbacks of your implementation and offer possible solutions if available.

## Pull request procedure

Follow these steps to submit a pull request:

1. Fork the `spey` repository.
2. Install pre-commit using `pip install pre-commit`
3. Go to `spey` main folder and type `pre-commit install`.
4. Open an issue and discuss the implementation with the developers.
5. Commit your changes to a feature branch on your fork and push all your changes.
6. Start a draft pull request and inform the developers about your progress.
7. Pull the ``main`` branch to ensure there are no conflicts with the current code developments.
8. Modify the appropriate section of
   `docs/releases/changelog-dev.md`.
9. Once complete, request a review from one of the maintainers.

#### Docstring style

Throughout the code following documentation style has been employed

```
{{! One-line RST Docstring Template }}
{{summaryPlaceholder}}

{{extendedSummaryPlaceholder}}

{{#parametersExist}}
Args:
{{/parametersExist}}
{{#args}}
    {{var}} (``{{typePlaceholder}}``): {{descriptionPlaceholder}}
{{/args}}
{{#kwargs}}
    {{var}} (``{{typePlaceholder}}``, default ``{{&default}}``): {{descriptionPlaceholder}}
{{/kwargs}}

{{#exceptionsExist}}
Raises:
{{/exceptionsExist}}
{{#exceptions}}
    ``{{type}}``: {{descriptionPlaceholder}}
{{/exceptions}}

{{#returnsExist}}
Returns:
{{/returnsExist}}
{{#returns}}
    ``{{typePlaceholder}}``:
    {{descriptionPlaceholder}}
{{/returns}}
```

This code can directly be imported into ``custom.mustache`` file and used within vscode as an auto docstring generator.
