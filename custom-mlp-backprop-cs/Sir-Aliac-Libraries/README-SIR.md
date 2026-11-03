# Why Not Use Sir Aliac Libraries?

## Short Explanation

Sir Aliac's libraries are **not used** in this project because they target **.NET Framework 3.5**, which **cannot run natively on macOS**.

## Details

- **Target Framework**: Sir Aliac's libraries were built for `.NET Framework 3.5` (legacy Windows-only framework)
- **Platform Incompatibility**: `.NET Framework 3.5` is Windows-only and does not run on macOS or Linux
- **Project Target**: This project targets **`.NET 10.0`** (modern, cross-platform)
- **Migration Path**: Porting the libraries to .NET 10.0 would require significant refactoring and is outside project scope

## Decision

All neural network math (Matrix, Layer, MLP, Activation) is implemented **from scratch** in this project under `src/NN/` to maintain:
- Cross-platform compatibility (macOS, Linux, Windows)
- Zero external ML dependencies
- Full control over the implementation for learning purposes