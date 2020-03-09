
Release process
***************

This is a reminder to the maintainers of how the release process is to be done.

* Overall based on: https://developer.skatelescope.org/ and in particular https://developer.skatelescope.org/en/latest/development/software_package_release_procedure.html
* Use semantic versioning: https://semver.org
* Follow the packaging process in: https://packaging.python.org/tutorials/packaging-projects/

Steps:
------

 * Ensure that the current master builds on GitLab: https://gitlab.com/timcornwell/rascil/pipelines
 * Decide whether a release is warranted and what semantic version number it should be: https://semver.org
 * Update setup.py for the new version number.
 * Update CHANGELOG.md for the relevant changes in this release.
 * Update README.md if appropriate
 * Tag the release e.g. v0.1.3 "Address hidden pip requirements"
 * Build the distribution and upload to PyPI::

        python3 setup.py sdist bdist_wheel
        python3 -m twine upload dist/*

 * Create a new virtualenv and try the install: pip3 install rascil::

        virtualenv test_env
        . test_env/bin/activate
        pip install rascil
        python3
        >>> import rascil
