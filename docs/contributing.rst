Contributing to Spey
====================

We are happy to accept contributions to ``spey`` via
`Pull Requests to our GitHub repository <https://github.com/SpeysideHEP/spey/pulls>`_.
You can begin this by forking the ``main`` repository.

Unless there is a minimal fix that does not require any discussion, please
always first `open an issue <https://github.com/SpeysideHEP/spey/issues/new/choose>`_
to discuss your request with the development team.

If the desired change is not limited to a couple of lines of code, please create
a draft pull request. This draft should detail the context of the change, its
description and the benefits of the implementation.

* If there is a change within the Python interface of the program, please proceed with standard tests and write extra tests if necessary.
* Please also make sure to add examples of using the new implementation.
* If your implementation has any drawbacks, these should be specified. Possible solutions should be offered, if any.


Pull request procedure
----------------------

Here are the steps to follow to make a pull request:

1. Fork the ``spey`` repository.
2. Open an issue and discuss the implementation with the developers.
3. Commit your changes to a feature branch on your fork and push all your changes there.
4. Start a draft pull request and inform the developers about your progress.
5. Pull the main branch to ensure there is no conflict with the current code developments.
6. Make sure you modified the appropriate section of  ``docs/releases/changelog-dev.md``.
7. Once you are done, request one of the maintainers to review your PR.

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