
### Expectations

flowMC is developed and maintained in my spare time and, while I try to be
responsive, I don't always get to every issue immediately. If it has been more
than a week or two, feel free to ping me (@kazewong) to try to get my attention. This is
subject to changes as the community grows.

### Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under
[Issues](https://github.com/kazewong/flowMC/issues). If you're unable to find an
open issue addressing the problem, [open a new
one](https://github.com/kazewong/flowMC/issues/new). Be sure to include a **title
and clear description**, as much relevant information as possible, and the
simplest possible **code sample** demonstrating the expected behavior that is
not occurring. Also label the issue with the bug label.

### Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch. Ensure the PR description clearly
describes the problem and solution. Include the relevant issue number if
applicable.

### Do you intend to add a new feature or change an existing feature?

Please follow the following principle when you are thinking about adding a new
feature or changing an existing feature:

1. The new feature should be able to take advantage of `jax.jit` whenever possible.
2. Light weight and modular implementation is preferred.
3. The core package only does sampling. If you have a concrete example that
   involves a complete analysis such as plotting and models, see the next
   contribution guide.

Suggestions for new features are welcome on [flowMC support
group](https://groups.google.com/u/1/g/flowmc). Note that features related to the
core algorithm are unlikely to be accepted since that may include a lot of
breaking changes.

### Do you intend to introduce an example or tutorial?

Open a new GitHub pull request with the example or tutorial. The example should
be self-contained and keep import from other packages to minimal. Leave the
case-specific analysis detail out. For more extensive tutorial, we encourage the
community to link the minimal example hosted on the flowMC documentation to
documentation from other packages.

### Do you have question about the code?

Do not open an issue. Instead, find whether there are already existing threads
on the [flowMC support group](https://groups.google.com/u/1/g/flowmc). If not,
please open a new conversation there.
