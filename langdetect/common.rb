#!/usr/bin/ruby -Ku

require 'mysql'
require 'optparse'

module LD
  LANGLIST = [
    # 日本語,中国語(繁体字),中国語(簡体字),韓国語,英語,フランス語,
    # イタリア語,スペイン語,ロシア語,アラビア語,ベトナム語,タイ語,
    "ja", "zh-CN", "zh-TW", "ko", "en", "fr", "it", "es", "ru", "ar", "vi", "th",
    # ドイツ語,ヒンディー語,ポルトガル語,インドネシア語,
    "de", "hi", "pt-PT", "id", 
    # "bn" # ベンガル語
    # "fa" # ペルシャ語
  ]

  ENTITIES = {
    "&#39;"=>"'",
    "&amp;"=>"&",
    "&gt;"=>">",
    "&lt;"=>"<",
    "&quot;"=>'"',
    "&raquo;"=>""
  }

  def self.optparser(additional_default = {})
    @opt = {
      :host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306,
      :model=>'model'
    }
    @opt.update(additional_default)

    parser = OptionParser.new
    parser.on('--host=VAL', String, 'database host') {|v| @opt[:host] = v }
    parser.on('--user=VAL', String, 'database user') {|v| @opt[:user] = v }
    parser.on('--password=VAL', String, 'database password') {|v| @opt[:passwd] = v }
    parser.on('--dbname=VAL', String, 'database name') {|v| @opt[:dbname] = v }
    parser.on('--port=VAL', Integer, 'database port') {|v| @opt[:port] = v }
    parser.on('-f VAL', String, 'model filename') {|v| @opt[:model] = v }
    parser
  end
  def self.opt
    @opt
  end

  def self.db_connect
    db = Mysql::init
    db.options Mysql::SET_CHARSET_NAME, 'utf8'
    db.real_connect @opt[:host], @opt[:user], @opt[:passwd], @opt[:dbname], @opt[:port]
    db
  end

  def self.decode_entity(st)
    st.gsub(/&[^ &]+?;/){|m| ENTITIES[m] || m}
  end

  def self.normalize(x)
    if x[0] <= 65
      " "
    elsif x =~ /^[\xd0-\xd1][\x80-\xbf]/      # Cyrillic
      "\xd0\x96"
    elsif x =~ /^[\xd8-\xd9][\x80-\xbf]/      # Arabic
      "\xd8\xa6"
    elsif x =~ /^\xe0[\xa4-\xa5][\x80-\xbf]/  # Devanagari
      "\xe0\xa4\x85"
    elsif x =~ /^\xe0[\xb8-\xb9][\x80-\xbf]/  # Thai
      "\xe0\xb9\x91"
    elsif x =~ /^\xe1[\xba-\xbb][\x80-\xbf]/  # Latin Extended Additional(Vietnam)
      "\xe1\xba\xa1"
    elsif x =~ /^\xe3[\x81-\x83][\x80-\xbf]/  # Hiragana / Katakana
      "\xe3\x81\x82"
    elsif x =~ /^\xea[\xb0-\xbf][\x80-\xbf]/  # Hangul Syllables 1
      "\xea\xb0\x80"
    elsif x =~ /^[\xeb-\xed][\x80-\xbf]{2}/   # Hangul Syllables 2
      "\xed\x9e\x98"
    else
      x
    end
  end

  class Ngram
    def initialize(n)
      @n = n
      clear
    end
    def clear
      @grams = [" "]
    end
    def append(x)
      clear if @grams[-1] == " "
      @grams << x
      @grams = @grams[-@n..-1] if @grams.length > @n
    end
    def get(n)
      return nil if @grams.length < n
      @grams[-n..-1].join
    end
    def each
      (1..@n).each do |n|
        x = get(n)
        yield x if x
      end
    end
  end
end

