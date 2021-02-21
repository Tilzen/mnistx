defmodule Mnistx do
  import Nx.Defn

  @default_defn_compiler ELXA

  @train_images "./lib/dataset/MNIST/train-images-idx3-ubyte.gz"
  @train_labels "./lib/dataset/MNIST/train-labels-idx1-ubyte.gz"

  @test_images "./lib/dataset/MNIST/t10k-images-idx3-ubyte.gz"
  @test_labels "./lib/dataset/MNIST/t10k-labels-idx1-ubyte.gz"

  ### Public API

  def start do
    train_images =
      @train_images
      |> read_dataset(:images)
      |> create_tensor(:images)

    train_labels =
      @train_labels
      |> read_dataset(:labels)
      |> create_tensor(:labels)

    zip = Enum.zip(train_images, train_labels) |> Enum.with_index()
    train(zip)
  end

  def train(zipped_images_labels) do
    for epoch <- 1..5, {{images, labels}, batch} <- zipped_images_labels, reduce: init_params() do
      params ->
        IO.puts "epoch #{epoch}, batch #{batch}"
        update(params, images, labels)
    end
  end

  ### Private API

  @doc """
  Performs a read of the image dataset and returns a tuple with the relevant
  information from the file.

  Returns `{number_images, number_rows, number_columns, images}`.

  ## Examples

    iex> Mnistx.read_dataset(path, :images)
    {60000, 28, 28,
     <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...>>}
  """
  defp read_dataset(path, :images) do
    <<_::32, number_images::32, number_rows::32, number_columns::32, images::binary>> =
      path
      |> Path.expand()
      |> File.read!()
      |> :zlib.gunzip()

    {number_images, number_rows, number_columns, images}
  end

  @doc """
  Performs a read of the labels dataset and returns a tuple with the relevant
  information from the file.

  Returns `{number_labels, labels}`.

  ## Examples

    iex> Mnistx.read_dataset(path, :labels)
    {60000,
     <<5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1, 2,
       4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, 5, ...>>}
  """
  defp read_dataset(path, :labels) do
    <<_::32, number_labels::32, labels::binary>> =
      path
      |> Path.expand()
      |> File.read!()
      |> :zlib.gunzip()

    {number_labels, labels}
  end

  @doc """
  Create and normalize the tensor for images.

  ## Examples

    iex> Mnistx.create_tensor({number_images, number_rows, number_columns, images}, :images)
  """
  defp create_tensor({number_images, number_rows, number_columns, images}, :images) do
    images
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({number_images, number_rows * number_columns}, names: [:batch, :input])
    |> Nx.divide(255)
    |> Nx.to_batched_list(30)
  end

  @doc """
  Create and normalize the tensor for labels.

  ## Examples

    iex> Mnistx.create_tensor({number_labels, labels}, :labels)
  """
  defp create_tensor({number_labels, labels}, :labels) do
    output_format = Nx.tensor(Enum.to_list(0..9))

    labels
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({number_labels, 1}, [:batch, :output])
    |> Nx.equal(output_format)
    |> Nx.to_batched_list(30)
  end

  ### Numerical definitions (Private)

  defnp init_params do
    weight_1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :hidden])
    bias_1 = Nx.random_normal({128}, 0.0, 0.1, names: [:hidden])
    weight_2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:hidden, :output])
    bias_2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {weight_1, bias_1, weight_2, bias_2}
  end

  @doc """
  The softmax function calculates the exponential of each element
  in the tensor, divided by the sum of the exponential of each
  element in the tensor.
  """
  defnp softmax(tensor) do
    Nx.exp(tensor) / Nx.sum(Nx.exp(tensor), axes: [:output], keep_axes: true)
  end

  defnp predict({weight_1, bias_1, weight_2, bias_2}, batch) do
    batch
    |> Nx.dot(weight_1)
    |> Nx.add(bias_1)
    |> Nx.logistic()
    |> Nx.dot(weight_2)
    |> Nx.add(bias_2)
    |> softmax()
  end

  defnp loss({weight_1, bias_1, weight_2, bias_2}, images, labels) do
    predictions = predict({weight_1, bias_1, weight_2, bias_2}, images)
    -Nx.sum(Nx.mean(Nx.log(predictions) * labels, axes: [:output]))
  end

  defnp update({weight_1, bias_1, weight_2, bias_2} = params, images, labels) do
    {gradient_w1, gradient_b1, gradient_w2, gradient_b2} =
      grad(params, loss(params, images, labels))

    {weight_1 - gradient_w1 * 0.01, bias_1 - gradient_b1 * 0.01,
     weight_2 - gradient_w2 * 0.01, bias_2 - gradient_b2 * 0.01}
  end
end
