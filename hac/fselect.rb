#!/usr/bin/ruby
# feature selection
# (cf. "Feature Selection and Document Clustering" http://www.csee.umbc.edu/cadip/2002Symposium/kogan.pdf )

require '../lib/infinitive.rb'
INF = Infinitive.new

require 'optparse'
opt = {:n_words=>1000, :type=>:q0, :stopwords=>true}
parser = OptionParser.new
parser.on('-n [VAL]', Integer) {|v| opt[:n_words] = v }
parser.on('-t [VAL]', [:q0, :q1]) {|v| opt[:type] = v }
parser.on('-s', 'exclude stop words') {|v| opt[:stopwords] = false }
parser.parse!(ARGV)


filename = ARGV[0] || 'corpus'
data = open(filename){|f| Marshal.load(f) }
docs = data[:docs]
terms = data[:terms]


stopwords = Hash.new(true)
<<STOPWORDS.scan(/[a-z]+/).each{|x| stopwords[INF.infinitive(x)] = false }
a about above across after afterwards again against all almost alone
along already also although always am among amongst amoungst amount an
and another any anyhow anyone anything anyway anywhere are around as
at back be became because become becomes becoming been before
beforehand behind being below beside besides between beyond bill both
bottom but by call can cannot cant co computer con could couldnt cry
de describe detail do done down due during each eg eight either eleven
else elsewhere empty enough etc even ever every everyone everything
everywhere except few fifteen fify fill find fire first five for
former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon
hers herself him himself his how however hundred i ie if in inc indeed
interest into is it its itself keep last latter latterly least less
ltd made many may me meanwhile might mill mine more moreover most
mostly move much must my myself name namely neither never nevertheless
next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours
ourselves out over own part per perhaps please put rather re same see
seem seemed seeming seems serious several she should show side since
sincere six sixty so some somehow someone something sometime sometimes
somewhere still such system take ten than that the their them
themselves then thence there thereafter thereby therefore therein
thereupon these they thick thin third this those though three through
throughout thru thus to together too top toward towards twelve twenty
two un under until up upon us very via was we well were what whatever
when whence whenever where whereafter whereas whereby wherein
whereupon wherever whether which while whither who whoever whole whom
whose why will with within without would yet you your yours yourself
yourselves
STOPWORDS




n_docs = docs.size

ev = []
terms.each do |term, rev_index|
  s1 = s2 = 0
  rev_index.each do |doc_id, freq|
    s1 += freq
    s2 += freq * freq
  end
  n = if opt[:type]==:q0 then n_docs else rev_index.size end
  v = s2 - (s1 * s1).to_f / n

  ev << [term, v] if opt[:stopwords] || stopwords[term]
end

ev = ev.sort{|a,b| b[1]<=>a[1]}[0..opt[:n_words]-1]

new_terms = {}
ev.each do |term, v|
  new_terms[term] = terms[term]
end

open("#{filename}.#{opt[:type]}", "w") do |f|
  Marshal.dump({:docs=>docs, :terms=>new_terms}, f)
end

puts "#{terms.size} => #{new_terms.size}"
#puts ev.map{|x| x[0]}.join(' ')
