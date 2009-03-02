#!/usr/bin/ruby
# feature selection

data = open(ARGV[0] || 'corpus'){|f| Marshal.load(f) }
docs = data[:docs]
terms = data[:terms]

n_docs = docs.size

ev = []
terms.each do |term, rev_index|
  s1 = s2 = 0
  rev_index.each do |doc_id, freq|
    s1 += freq
    s2 += freq * freq
  end
  v = s2 - (s1 * s1).to_f / rev_index.size

  ev << [term, v]
end

ev.sort{|a,b| b[1]<=>a[1]}.each do |term, v|
  puts "#{term},#{v}"
end


