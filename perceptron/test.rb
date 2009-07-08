#!/usr/bin/ruby
if ARGV.length < 2
  puts "#$0 testfile modelfile"
  exit 1
end

w = open(ARGV[1]){|f| Marshal.load(f) }

# load test data
data = []
open(ARGV[0]) do |f|
  while line = f.gets
    features = line.split
    sign = features.shift.to_i
    map = Hash.new
    features.each do |feature|
      if feature =~ /^([0-9]+):([\+\-]?[0-9\.]+)$/
        map[$1.to_i] = $2.to_f
      end
    end
    data << [map, sign]
  end
end

result = Array.new(4, 0)
data.each do |x, t|
  x[w.size-1] = 1 # bias
  s = 0
  x.each do |i, x_i|
    s += w[i] * x_i if i < w.size
  end
  result[(t>0?2:0)+(s>0?1:0)] += 1
end

puts "Accuracy #{((result[3]+result[0]).to_f/data.size*100000).round/1000.0}% (#{result[3]+result[0]}/#{data.size})"
puts "(Answer, Predict): (p,p):#{result[3]} (p,n):#{result[2]} (n,p):#{result[1]} (n,n):#{result[0]}"



