#!/usr/bin/ruby

require 'zlib'
require "neural.rb"

n_rows = n_cols = nil
images = []
labels = []
Zlib::GzipReader.open('train-images-idx3-ubyte.gz') do |f|
  magic, n_images = f.read(8).unpack('N2')
  raise 'This is not MNIST image file' if magic != 2051
  n_rows, n_cols = f.read(8).unpack('N2')
  n_images.times do
    images << f.read(n_rows * n_cols)
  end
end

Zlib::GzipReader.open('train-labels-idx1-ubyte.gz') do |f|
  magic, n_labels = f.read(8).unpack('N2')
  raise 'This is not MNIST label file' if magic != 2049
  labels = f.read(n_labels).unpack('C*')
end

# output pgm
def output_pgm(filename, images, n_rows, n_cols, n_width, n_height)
  open(filename, "wb") do |f|
    f.printf("P5 %d %d %d ", n_rows*n_width, n_cols*n_height, 0xff)
    offset = 0
    buf = ""
    n_height.times do
      n_cols.times do |y|
        n_width.times do |idx|
          st = images[offset + idx][y * n_rows, n_rows].unpack('C*').map{|p| 0xff - p }.pack("C*")
          #p images[offset + idx][y * n_rows, n_rows].unpack('C*') if st.length != 28
          buf << st
        end
      end
      offset += n_width
    end
    f.puts buf
  end
end
#output_pgm "mnist.pgm", images, n_rows, n_cols, 300, 200

# units
in_units = (1..(28*28)).map{|i| Unit.new("x#{i}")}
bias = [BiasUnit.new("1")]
hiddenunits = (1..300).map{|i| TanhUnit.new("z#{i}")}
out_unit = (1..10).map{|i| SigUnit.new("y#{i}")}

# network
network = Network.new
network.in  = in_units
network.link in_units + bias, hiddenunits
network.link hiddenunits + bias, out_unit
network.out = out_unit

eta = 0.1
  images.each_with_index do |image, idx|
    image = image.unpack('C*')
    target = [0]*10
    target[labels[idx]] = 1
    grad = network.gradient_E_backward(image, target)
    network.weights.descent eta, grad
    puts idx
  end



