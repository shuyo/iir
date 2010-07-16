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

  def self.optparser
    @opt = {
      :host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306,
      :model=>'model'
    }

    parser = OptionParser.new
    parser.on('--host=VAL', String, 'database host') {|v| @opt[:host] = v }
    parser.on('--user=VAL', String, 'database user') {|v| @opt[:user] = v }
    parser.on('--password=VAL', String, 'database password') {|v| @opt[:passwd] = v }
    parser.on('--dbname=VAL', String, 'database name') {|v| @opt[:dbname] = v }
    parser.on('--port=VAL', Integer, 'database port') {|v| @opt[:port] = v }
    parser.on('-f VAL', String, 'model filename') {|v| @opt[:model] = v }
    parser
  end
  def self.model_filename
    @opt[:model]
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
end

