defmodule Mnistx.MixProject do
  use Mix.Project

  def project do
    [
      app: :mnistx,
      version: "0.1.0",
      elixir: "~> 1.10",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx", override: true}
    ]
  end
end
