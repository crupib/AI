/**
 * Autogenerated by Thrift Compiler (0.9.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
package com.datastax.demo.portfolio;

import org.apache.commons.lang.builder.HashCodeBuilder;
import org.apache.thrift.scheme.IScheme;
import org.apache.thrift.scheme.SchemeFactory;
import org.apache.thrift.scheme.StandardScheme;

import org.apache.thrift.scheme.TupleScheme;
import org.apache.thrift.protocol.TTupleProtocol;
import org.apache.thrift.protocol.TProtocolException;
import org.apache.thrift.EncodingUtils;
import org.apache.thrift.TException;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.EnumMap;
import java.util.Set;
import java.util.HashSet;
import java.util.EnumSet;
import java.util.Collections;
import java.util.BitSet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Portfolio implements org.apache.thrift.TBase<Portfolio, Portfolio._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("Portfolio");

  private static final org.apache.thrift.protocol.TField NAME_FIELD_DESC = new org.apache.thrift.protocol.TField("name", org.apache.thrift.protocol.TType.STRING, (short)1);
  private static final org.apache.thrift.protocol.TField CONSTITUENTS_FIELD_DESC = new org.apache.thrift.protocol.TField("constituents", org.apache.thrift.protocol.TType.LIST, (short)2);
  private static final org.apache.thrift.protocol.TField BASIS_FIELD_DESC = new org.apache.thrift.protocol.TField("basis", org.apache.thrift.protocol.TType.DOUBLE, (short)3);
  private static final org.apache.thrift.protocol.TField PRICE_FIELD_DESC = new org.apache.thrift.protocol.TField("price", org.apache.thrift.protocol.TType.DOUBLE, (short)4);
  private static final org.apache.thrift.protocol.TField LARGEST_10DAY_LOSS_FIELD_DESC = new org.apache.thrift.protocol.TField("largest_10day_loss", org.apache.thrift.protocol.TType.DOUBLE, (short)5);
  private static final org.apache.thrift.protocol.TField LARGEST_10DAY_LOSS_DATE_FIELD_DESC = new org.apache.thrift.protocol.TField("largest_10day_loss_date", org.apache.thrift.protocol.TType.STRING, (short)6);
  private static final org.apache.thrift.protocol.TField HIST_PRICES_FIELD_DESC = new org.apache.thrift.protocol.TField("hist_prices", org.apache.thrift.protocol.TType.LIST, (short)7);

  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new PortfolioStandardSchemeFactory());
    schemes.put(TupleScheme.class, new PortfolioTupleSchemeFactory());
  }

  public String name; // required
  public List<Position> constituents; // required
  public double basis; // required
  public double price; // required
  public double largest_10day_loss; // required
  public String largest_10day_loss_date; // required
  public List<Double> hist_prices; // required

  /** The set of fields this struct contains, along with convenience methods for finding and manipulating them. */
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    NAME((short)1, "name"),
    CONSTITUENTS((short)2, "constituents"),
    BASIS((short)3, "basis"),
    PRICE((short)4, "price"),
    LARGEST_10DAY_LOSS((short)5, "largest_10day_loss"),
    LARGEST_10DAY_LOSS_DATE((short)6, "largest_10day_loss_date"),
    HIST_PRICES((short)7, "hist_prices");

    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();

    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, or null if its not found.
     */
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
        case 1: // NAME
          return NAME;
        case 2: // CONSTITUENTS
          return CONSTITUENTS;
        case 3: // BASIS
          return BASIS;
        case 4: // PRICE
          return PRICE;
        case 5: // LARGEST_10DAY_LOSS
          return LARGEST_10DAY_LOSS;
        case 6: // LARGEST_10DAY_LOSS_DATE
          return LARGEST_10DAY_LOSS_DATE;
        case 7: // HIST_PRICES
          return HIST_PRICES;
        default:
          return null;
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, throwing an exception
     * if it is not found.
     */
    public static _Fields findByThriftIdOrThrow(int fieldId) {
      _Fields fields = findByThriftId(fieldId);
      if (fields == null) throw new IllegalArgumentException("Field " + fieldId + " doesn't exist!");
      return fields;
    }

    /**
     * Find the _Fields constant that matches name, or null if its not found.
     */
    public static _Fields findByName(String name) {
      return byName.get(name);
    }

    private final short _thriftId;
    private final String _fieldName;

    _Fields(short thriftId, String fieldName) {
      _thriftId = thriftId;
      _fieldName = fieldName;
    }

    public short getThriftFieldId() {
      return _thriftId;
    }

    public String getFieldName() {
      return _fieldName;
    }
  }

  // isset id assignments
  private static final int __BASIS_ISSET_ID = 0;
  private static final int __PRICE_ISSET_ID = 1;
  private static final int __LARGEST_10DAY_LOSS_ISSET_ID = 2;
  private byte __isset_bitfield = 0;
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.NAME, new org.apache.thrift.meta_data.FieldMetaData("name", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
    tmpMap.put(_Fields.CONSTITUENTS, new org.apache.thrift.meta_data.FieldMetaData("constituents", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, Position.class))));
    tmpMap.put(_Fields.BASIS, new org.apache.thrift.meta_data.FieldMetaData("basis", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    tmpMap.put(_Fields.PRICE, new org.apache.thrift.meta_data.FieldMetaData("price", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    tmpMap.put(_Fields.LARGEST_10DAY_LOSS, new org.apache.thrift.meta_data.FieldMetaData("largest_10day_loss", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    tmpMap.put(_Fields.LARGEST_10DAY_LOSS_DATE, new org.apache.thrift.meta_data.FieldMetaData("largest_10day_loss_date", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
    tmpMap.put(_Fields.HIST_PRICES, new org.apache.thrift.meta_data.FieldMetaData("hist_prices", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE))));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(Portfolio.class, metaDataMap);
  }

  public Portfolio() {
  }

  public Portfolio(
    String name,
    List<Position> constituents,
    double basis,
    double price,
    double largest_10day_loss,
    String largest_10day_loss_date,
    List<Double> hist_prices)
  {
    this();
    this.name = name;
    this.constituents = constituents;
    this.basis = basis;
    setBasisIsSet(true);
    this.price = price;
    setPriceIsSet(true);
    this.largest_10day_loss = largest_10day_loss;
    setLargest_10day_lossIsSet(true);
    this.largest_10day_loss_date = largest_10day_loss_date;
    this.hist_prices = hist_prices;
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public Portfolio(Portfolio other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetName()) {
      this.name = other.name;
    }
    if (other.isSetConstituents()) {
      List<Position> __this__constituents = new ArrayList<Position>();
      for (Position other_element : other.constituents) {
        __this__constituents.add(new Position(other_element));
      }
      this.constituents = __this__constituents;
    }
    this.basis = other.basis;
    this.price = other.price;
    this.largest_10day_loss = other.largest_10day_loss;
    if (other.isSetLargest_10day_loss_date()) {
      this.largest_10day_loss_date = other.largest_10day_loss_date;
    }
    if (other.isSetHist_prices()) {
      List<Double> __this__hist_prices = new ArrayList<Double>();
      for (Double other_element : other.hist_prices) {
        __this__hist_prices.add(other_element);
      }
      this.hist_prices = __this__hist_prices;
    }
  }

  public Portfolio deepCopy() {
    return new Portfolio(this);
  }

  @Override
  public void clear() {
    this.name = null;
    this.constituents = null;
    setBasisIsSet(false);
    this.basis = 0.0;
    setPriceIsSet(false);
    this.price = 0.0;
    setLargest_10day_lossIsSet(false);
    this.largest_10day_loss = 0.0;
    this.largest_10day_loss_date = null;
    this.hist_prices = null;
  }

  public String getName() {
    return this.name;
  }

  public Portfolio setName(String name) {
    this.name = name;
    return this;
  }

  public void unsetName() {
    this.name = null;
  }

  /** Returns true if field name is set (has been assigned a value) and false otherwise */
  public boolean isSetName() {
    return this.name != null;
  }

  public void setNameIsSet(boolean value) {
    if (!value) {
      this.name = null;
    }
  }

  public int getConstituentsSize() {
    return (this.constituents == null) ? 0 : this.constituents.size();
  }

  public java.util.Iterator<Position> getConstituentsIterator() {
    return (this.constituents == null) ? null : this.constituents.iterator();
  }

  public void addToConstituents(Position elem) {
    if (this.constituents == null) {
      this.constituents = new ArrayList<Position>();
    }
    this.constituents.add(elem);
  }

  public List<Position> getConstituents() {
    return this.constituents;
  }

  public Portfolio setConstituents(List<Position> constituents) {
    this.constituents = constituents;
    return this;
  }

  public void unsetConstituents() {
    this.constituents = null;
  }

  /** Returns true if field constituents is set (has been assigned a value) and false otherwise */
  public boolean isSetConstituents() {
    return this.constituents != null;
  }

  public void setConstituentsIsSet(boolean value) {
    if (!value) {
      this.constituents = null;
    }
  }

  public double getBasis() {
    return this.basis;
  }

  public Portfolio setBasis(double basis) {
    this.basis = basis;
    setBasisIsSet(true);
    return this;
  }

  public void unsetBasis() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __BASIS_ISSET_ID);
  }

  /** Returns true if field basis is set (has been assigned a value) and false otherwise */
  public boolean isSetBasis() {
    return EncodingUtils.testBit(__isset_bitfield, __BASIS_ISSET_ID);
  }

  public void setBasisIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __BASIS_ISSET_ID, value);
  }

  public double getPrice() {
    return this.price;
  }

  public Portfolio setPrice(double price) {
    this.price = price;
    setPriceIsSet(true);
    return this;
  }

  public void unsetPrice() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __PRICE_ISSET_ID);
  }

  /** Returns true if field price is set (has been assigned a value) and false otherwise */
  public boolean isSetPrice() {
    return EncodingUtils.testBit(__isset_bitfield, __PRICE_ISSET_ID);
  }

  public void setPriceIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __PRICE_ISSET_ID, value);
  }

  public double getLargest_10day_loss() {
    return this.largest_10day_loss;
  }

  public Portfolio setLargest_10day_loss(double largest_10day_loss) {
    this.largest_10day_loss = largest_10day_loss;
    setLargest_10day_lossIsSet(true);
    return this;
  }

  public void unsetLargest_10day_loss() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __LARGEST_10DAY_LOSS_ISSET_ID);
  }

  /** Returns true if field largest_10day_loss is set (has been assigned a value) and false otherwise */
  public boolean isSetLargest_10day_loss() {
    return EncodingUtils.testBit(__isset_bitfield, __LARGEST_10DAY_LOSS_ISSET_ID);
  }

  public void setLargest_10day_lossIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __LARGEST_10DAY_LOSS_ISSET_ID, value);
  }

  public String getLargest_10day_loss_date() {
    return this.largest_10day_loss_date;
  }

  public Portfolio setLargest_10day_loss_date(String largest_10day_loss_date) {
    this.largest_10day_loss_date = largest_10day_loss_date;
    return this;
  }

  public void unsetLargest_10day_loss_date() {
    this.largest_10day_loss_date = null;
  }

  /** Returns true if field largest_10day_loss_date is set (has been assigned a value) and false otherwise */
  public boolean isSetLargest_10day_loss_date() {
    return this.largest_10day_loss_date != null;
  }

  public void setLargest_10day_loss_dateIsSet(boolean value) {
    if (!value) {
      this.largest_10day_loss_date = null;
    }
  }

  public int getHist_pricesSize() {
    return (this.hist_prices == null) ? 0 : this.hist_prices.size();
  }

  public java.util.Iterator<Double> getHist_pricesIterator() {
    return (this.hist_prices == null) ? null : this.hist_prices.iterator();
  }

  public void addToHist_prices(double elem) {
    if (this.hist_prices == null) {
      this.hist_prices = new ArrayList<Double>();
    }
    this.hist_prices.add(elem);
  }

  public List<Double> getHist_prices() {
    return this.hist_prices;
  }

  public Portfolio setHist_prices(List<Double> hist_prices) {
    this.hist_prices = hist_prices;
    return this;
  }

  public void unsetHist_prices() {
    this.hist_prices = null;
  }

  /** Returns true if field hist_prices is set (has been assigned a value) and false otherwise */
  public boolean isSetHist_prices() {
    return this.hist_prices != null;
  }

  public void setHist_pricesIsSet(boolean value) {
    if (!value) {
      this.hist_prices = null;
    }
  }

  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case NAME:
      if (value == null) {
        unsetName();
      } else {
        setName((String)value);
      }
      break;

    case CONSTITUENTS:
      if (value == null) {
        unsetConstituents();
      } else {
        setConstituents((List<Position>)value);
      }
      break;

    case BASIS:
      if (value == null) {
        unsetBasis();
      } else {
        setBasis((Double)value);
      }
      break;

    case PRICE:
      if (value == null) {
        unsetPrice();
      } else {
        setPrice((Double)value);
      }
      break;

    case LARGEST_10DAY_LOSS:
      if (value == null) {
        unsetLargest_10day_loss();
      } else {
        setLargest_10day_loss((Double)value);
      }
      break;

    case LARGEST_10DAY_LOSS_DATE:
      if (value == null) {
        unsetLargest_10day_loss_date();
      } else {
        setLargest_10day_loss_date((String)value);
      }
      break;

    case HIST_PRICES:
      if (value == null) {
        unsetHist_prices();
      } else {
        setHist_prices((List<Double>)value);
      }
      break;

    }
  }

  public Object getFieldValue(_Fields field) {
    switch (field) {
    case NAME:
      return getName();

    case CONSTITUENTS:
      return getConstituents();

    case BASIS:
      return Double.valueOf(getBasis());

    case PRICE:
      return Double.valueOf(getPrice());

    case LARGEST_10DAY_LOSS:
      return Double.valueOf(getLargest_10day_loss());

    case LARGEST_10DAY_LOSS_DATE:
      return getLargest_10day_loss_date();

    case HIST_PRICES:
      return getHist_prices();

    }
    throw new IllegalStateException();
  }

  /** Returns true if field corresponding to fieldID is set (has been assigned a value) and false otherwise */
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }

    switch (field) {
    case NAME:
      return isSetName();
    case CONSTITUENTS:
      return isSetConstituents();
    case BASIS:
      return isSetBasis();
    case PRICE:
      return isSetPrice();
    case LARGEST_10DAY_LOSS:
      return isSetLargest_10day_loss();
    case LARGEST_10DAY_LOSS_DATE:
      return isSetLargest_10day_loss_date();
    case HIST_PRICES:
      return isSetHist_prices();
    }
    throw new IllegalStateException();
  }

  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof Portfolio)
      return this.equals((Portfolio)that);
    return false;
  }

  public boolean equals(Portfolio that) {
    if (that == null)
      return false;

    boolean this_present_name = true && this.isSetName();
    boolean that_present_name = true && that.isSetName();
    if (this_present_name || that_present_name) {
      if (!(this_present_name && that_present_name))
        return false;
      if (!this.name.equals(that.name))
        return false;
    }

    boolean this_present_constituents = true && this.isSetConstituents();
    boolean that_present_constituents = true && that.isSetConstituents();
    if (this_present_constituents || that_present_constituents) {
      if (!(this_present_constituents && that_present_constituents))
        return false;
      if (!this.constituents.equals(that.constituents))
        return false;
    }

    boolean this_present_basis = true;
    boolean that_present_basis = true;
    if (this_present_basis || that_present_basis) {
      if (!(this_present_basis && that_present_basis))
        return false;
      if (this.basis != that.basis)
        return false;
    }

    boolean this_present_price = true;
    boolean that_present_price = true;
    if (this_present_price || that_present_price) {
      if (!(this_present_price && that_present_price))
        return false;
      if (this.price != that.price)
        return false;
    }

    boolean this_present_largest_10day_loss = true;
    boolean that_present_largest_10day_loss = true;
    if (this_present_largest_10day_loss || that_present_largest_10day_loss) {
      if (!(this_present_largest_10day_loss && that_present_largest_10day_loss))
        return false;
      if (this.largest_10day_loss != that.largest_10day_loss)
        return false;
    }

    boolean this_present_largest_10day_loss_date = true && this.isSetLargest_10day_loss_date();
    boolean that_present_largest_10day_loss_date = true && that.isSetLargest_10day_loss_date();
    if (this_present_largest_10day_loss_date || that_present_largest_10day_loss_date) {
      if (!(this_present_largest_10day_loss_date && that_present_largest_10day_loss_date))
        return false;
      if (!this.largest_10day_loss_date.equals(that.largest_10day_loss_date))
        return false;
    }

    boolean this_present_hist_prices = true && this.isSetHist_prices();
    boolean that_present_hist_prices = true && that.isSetHist_prices();
    if (this_present_hist_prices || that_present_hist_prices) {
      if (!(this_present_hist_prices && that_present_hist_prices))
        return false;
      if (!this.hist_prices.equals(that.hist_prices))
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    HashCodeBuilder builder = new HashCodeBuilder();

    boolean present_name = true && (isSetName());
    builder.append(present_name);
    if (present_name)
      builder.append(name);

    boolean present_constituents = true && (isSetConstituents());
    builder.append(present_constituents);
    if (present_constituents)
      builder.append(constituents);

    boolean present_basis = true;
    builder.append(present_basis);
    if (present_basis)
      builder.append(basis);

    boolean present_price = true;
    builder.append(present_price);
    if (present_price)
      builder.append(price);

    boolean present_largest_10day_loss = true;
    builder.append(present_largest_10day_loss);
    if (present_largest_10day_loss)
      builder.append(largest_10day_loss);

    boolean present_largest_10day_loss_date = true && (isSetLargest_10day_loss_date());
    builder.append(present_largest_10day_loss_date);
    if (present_largest_10day_loss_date)
      builder.append(largest_10day_loss_date);

    boolean present_hist_prices = true && (isSetHist_prices());
    builder.append(present_hist_prices);
    if (present_hist_prices)
      builder.append(hist_prices);

    return builder.toHashCode();
  }

  public int compareTo(Portfolio other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;
    Portfolio typedOther = (Portfolio)other;

    lastComparison = Boolean.valueOf(isSetName()).compareTo(typedOther.isSetName());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetName()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.name, typedOther.name);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetConstituents()).compareTo(typedOther.isSetConstituents());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetConstituents()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.constituents, typedOther.constituents);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetBasis()).compareTo(typedOther.isSetBasis());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetBasis()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.basis, typedOther.basis);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetPrice()).compareTo(typedOther.isSetPrice());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetPrice()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.price, typedOther.price);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetLargest_10day_loss()).compareTo(typedOther.isSetLargest_10day_loss());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetLargest_10day_loss()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.largest_10day_loss, typedOther.largest_10day_loss);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetLargest_10day_loss_date()).compareTo(typedOther.isSetLargest_10day_loss_date());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetLargest_10day_loss_date()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.largest_10day_loss_date, typedOther.largest_10day_loss_date);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetHist_prices()).compareTo(typedOther.isSetHist_prices());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetHist_prices()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.hist_prices, typedOther.hist_prices);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    return 0;
  }

  public _Fields fieldForId(int fieldId) {
    return _Fields.findByThriftId(fieldId);
  }

  public void read(org.apache.thrift.protocol.TProtocol iprot) throws org.apache.thrift.TException {
    schemes.get(iprot.getScheme()).getScheme().read(iprot, this);
  }

  public void write(org.apache.thrift.protocol.TProtocol oprot) throws org.apache.thrift.TException {
    schemes.get(oprot.getScheme()).getScheme().write(oprot, this);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Portfolio(");
    boolean first = true;

    sb.append("name:");
    if (this.name == null) {
      sb.append("null");
    } else {
      sb.append(this.name);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("constituents:");
    if (this.constituents == null) {
      sb.append("null");
    } else {
      sb.append(this.constituents);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("basis:");
    sb.append(this.basis);
    first = false;
    if (!first) sb.append(", ");
    sb.append("price:");
    sb.append(this.price);
    first = false;
    if (!first) sb.append(", ");
    sb.append("largest_10day_loss:");
    sb.append(this.largest_10day_loss);
    first = false;
    if (!first) sb.append(", ");
    sb.append("largest_10day_loss_date:");
    if (this.largest_10day_loss_date == null) {
      sb.append("null");
    } else {
      sb.append(this.largest_10day_loss_date);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("hist_prices:");
    if (this.hist_prices == null) {
      sb.append("null");
    } else {
      sb.append(this.hist_prices);
    }
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws org.apache.thrift.TException {
    // check for required fields
    // check for sub-struct validity
  }

  private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    try {
      write(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(out)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
    try {
      // it doesn't seem like you should have to do this, but java serialization is wacky, and doesn't call the default constructor.
      __isset_bitfield = 0;
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private static class PortfolioStandardSchemeFactory implements SchemeFactory {
    public PortfolioStandardScheme getScheme() {
      return new PortfolioStandardScheme();
    }
  }

  private static class PortfolioStandardScheme extends StandardScheme<Portfolio> {

    public void read(org.apache.thrift.protocol.TProtocol iprot, Portfolio struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
          case 1: // NAME
            if (schemeField.type == org.apache.thrift.protocol.TType.STRING) {
              struct.name = iprot.readString();
              struct.setNameIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 2: // CONSTITUENTS
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list0 = iprot.readListBegin();
                struct.constituents = new ArrayList<Position>(_list0.size);
                for (int _i1 = 0; _i1 < _list0.size; ++_i1)
                {
                  Position _elem2; // required
                  _elem2 = new Position();
                  _elem2.read(iprot);
                  struct.constituents.add(_elem2);
                }
                iprot.readListEnd();
              }
              struct.setConstituentsIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 3: // BASIS
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.basis = iprot.readDouble();
              struct.setBasisIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 4: // PRICE
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.price = iprot.readDouble();
              struct.setPriceIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 5: // LARGEST_10DAY_LOSS
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.largest_10day_loss = iprot.readDouble();
              struct.setLargest_10day_lossIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 6: // LARGEST_10DAY_LOSS_DATE
            if (schemeField.type == org.apache.thrift.protocol.TType.STRING) {
              struct.largest_10day_loss_date = iprot.readString();
              struct.setLargest_10day_loss_dateIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 7: // HIST_PRICES
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list3 = iprot.readListBegin();
                struct.hist_prices = new ArrayList<Double>(_list3.size);
                for (int _i4 = 0; _i4 < _list3.size; ++_i4)
                {
                  double _elem5; // required
                  _elem5 = iprot.readDouble();
                  struct.hist_prices.add(_elem5);
                }
                iprot.readListEnd();
              }
              struct.setHist_pricesIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          default:
            org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
        }
        iprot.readFieldEnd();
      }
      iprot.readStructEnd();

      // check for required fields of primitive type, which can't be checked in the validate method
      struct.validate();
    }

    public void write(org.apache.thrift.protocol.TProtocol oprot, Portfolio struct) throws org.apache.thrift.TException {
      struct.validate();

      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.name != null) {
        oprot.writeFieldBegin(NAME_FIELD_DESC);
        oprot.writeString(struct.name);
        oprot.writeFieldEnd();
      }
      if (struct.constituents != null) {
        oprot.writeFieldBegin(CONSTITUENTS_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, struct.constituents.size()));
          for (Position _iter6 : struct.constituents)
          {
            _iter6.write(oprot);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldBegin(BASIS_FIELD_DESC);
      oprot.writeDouble(struct.basis);
      oprot.writeFieldEnd();
      oprot.writeFieldBegin(PRICE_FIELD_DESC);
      oprot.writeDouble(struct.price);
      oprot.writeFieldEnd();
      oprot.writeFieldBegin(LARGEST_10DAY_LOSS_FIELD_DESC);
      oprot.writeDouble(struct.largest_10day_loss);
      oprot.writeFieldEnd();
      if (struct.largest_10day_loss_date != null) {
        oprot.writeFieldBegin(LARGEST_10DAY_LOSS_DATE_FIELD_DESC);
        oprot.writeString(struct.largest_10day_loss_date);
        oprot.writeFieldEnd();
      }
      if (struct.hist_prices != null) {
        oprot.writeFieldBegin(HIST_PRICES_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.DOUBLE, struct.hist_prices.size()));
          for (double _iter7 : struct.hist_prices)
          {
            oprot.writeDouble(_iter7);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }

  }

  private static class PortfolioTupleSchemeFactory implements SchemeFactory {
    public PortfolioTupleScheme getScheme() {
      return new PortfolioTupleScheme();
    }
  }

  private static class PortfolioTupleScheme extends TupleScheme<Portfolio> {

    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, Portfolio struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      BitSet optionals = new BitSet();
      if (struct.isSetName()) {
        optionals.set(0);
      }
      if (struct.isSetConstituents()) {
        optionals.set(1);
      }
      if (struct.isSetBasis()) {
        optionals.set(2);
      }
      if (struct.isSetPrice()) {
        optionals.set(3);
      }
      if (struct.isSetLargest_10day_loss()) {
        optionals.set(4);
      }
      if (struct.isSetLargest_10day_loss_date()) {
        optionals.set(5);
      }
      if (struct.isSetHist_prices()) {
        optionals.set(6);
      }
      oprot.writeBitSet(optionals, 7);
      if (struct.isSetName()) {
        oprot.writeString(struct.name);
      }
      if (struct.isSetConstituents()) {
        {
          oprot.writeI32(struct.constituents.size());
          for (Position _iter8 : struct.constituents)
          {
            _iter8.write(oprot);
          }
        }
      }
      if (struct.isSetBasis()) {
        oprot.writeDouble(struct.basis);
      }
      if (struct.isSetPrice()) {
        oprot.writeDouble(struct.price);
      }
      if (struct.isSetLargest_10day_loss()) {
        oprot.writeDouble(struct.largest_10day_loss);
      }
      if (struct.isSetLargest_10day_loss_date()) {
        oprot.writeString(struct.largest_10day_loss_date);
      }
      if (struct.isSetHist_prices()) {
        {
          oprot.writeI32(struct.hist_prices.size());
          for (double _iter9 : struct.hist_prices)
          {
            oprot.writeDouble(_iter9);
          }
        }
      }
    }

    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, Portfolio struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      BitSet incoming = iprot.readBitSet(7);
      if (incoming.get(0)) {
        struct.name = iprot.readString();
        struct.setNameIsSet(true);
      }
      if (incoming.get(1)) {
        {
          org.apache.thrift.protocol.TList _list10 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, iprot.readI32());
          struct.constituents = new ArrayList<Position>(_list10.size);
          for (int _i11 = 0; _i11 < _list10.size; ++_i11)
          {
            Position _elem12; // required
            _elem12 = new Position();
            _elem12.read(iprot);
            struct.constituents.add(_elem12);
          }
        }
        struct.setConstituentsIsSet(true);
      }
      if (incoming.get(2)) {
        struct.basis = iprot.readDouble();
        struct.setBasisIsSet(true);
      }
      if (incoming.get(3)) {
        struct.price = iprot.readDouble();
        struct.setPriceIsSet(true);
      }
      if (incoming.get(4)) {
        struct.largest_10day_loss = iprot.readDouble();
        struct.setLargest_10day_lossIsSet(true);
      }
      if (incoming.get(5)) {
        struct.largest_10day_loss_date = iprot.readString();
        struct.setLargest_10day_loss_dateIsSet(true);
      }
      if (incoming.get(6)) {
        {
          org.apache.thrift.protocol.TList _list13 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.DOUBLE, iprot.readI32());
          struct.hist_prices = new ArrayList<Double>(_list13.size);
          for (int _i14 = 0; _i14 < _list13.size; ++_i14)
          {
            double _elem15; // required
            _elem15 = iprot.readDouble();
            struct.hist_prices.add(_elem15);
          }
        }
        struct.setHist_pricesIsSet(true);
      }
    }
  }

}

