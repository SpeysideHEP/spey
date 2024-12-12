Contributing to Spey
====================

We welcome contributions to ``spey`` via
`Pull Requests to our GitHub repository <https://github.com/SpeysideHEP/spey/pulls>`_.
To get started, fork the ``main`` repository.

For anything beyond minimal fixes that do not require discussion, please first `open an issue <https://github.com/SpeysideHEP/spey/issues/new/choose>`_
to discuss your request with the development team.

If your proposed changes are more extensive than a few lines of code, please create a draft pull request.
This draft should include the context of the change, a description, and the benefits of the implementation.

* For changes within the Python interface of the program, please run standard tests and write additional tests if necessary.
* Ensure you add examples demonstrating the new implementation.
* Specify any drawbacks of your implementation and offer possible solutions if available.


Pull request procedure
----------------------

Follow these steps to submit a pull request:

1. Fork the ``spey`` repository.
2. Open an issue and discuss the implementation with the developers.
3. Commit your changes to a feature branch on your fork and push all your changes.
4. Start a draft pull request and inform the developers about your progress.
5. Pull the ``main`` branch to ensure there are no conflicts with the current code developments.
6. Modify the appropriate section of  ``docs/releases/changelog-dev.md``.
7. Once complete, request a review from one of the maintainers.

Docstring style
~~~~~~~~~~~~~~~

Throughout the code, the following documentation style has been employed

.. code-block::

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

This code can directly be imported into ``custom.mustache`` file and used within
vscode as an auto docstring generator.
