from OpenSSL import crypto


def generate_self_signed_certs(
        email_address="emailAddress",
        common_name="commonName",
        country_name="NT",
        locality_name="localityName",
        state_or_province_name="stateOrProvinceName",
        organization_name="organizationName",
        organization_unit_name="organizationUnitName",
        serial_number=0,
        validity_start_in_seconds=0,
        validity_end_in_seconds=10 * 365 * 24 * 60 * 60,
        out_key_file="private.key",
        out_cert_file="self-signed.crt"):
    """ Creates a self_signed certificate

        Parameters:
            email_address: certificate email_address
            common_name: certificate common name
            country_name: certificate country name
            locality_name: certificate locality name
            state_or_province_name: certificate state/province name
            organization_name: certificate organization name
            organization_unit_name: certificate organization unit name
            serial_number: certificate serial number
            validity_start_in_seconds: certificate start validity in seconds
            validity_end_in_seconds: certificate end of validity in seconds
            out_key_file: output file for the certificate private key
            out_cert_file: output file for certificate

    """
    # can look at generated file using openssl:
    # openssl x509 -inform pem -in selfsigned.crt -noout -text
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = country_name
    cert.get_subject().ST = state_or_province_name
    cert.get_subject().L = locality_name
    cert.get_subject().O = organization_name
    cert.get_subject().OU = organization_unit_name
    cert.get_subject().CN = common_name
    cert.get_subject().emailAddress = email_address
    cert.set_serial_number(serial_number)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(validity_end_in_seconds)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha512')
    with open(out_cert_file, "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
    with open(out_key_file, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))


def generate_certs(
        email_address="emailAddress",
        common_name="None",
        country_name="NT",
        locality_name="localityName",
        state_or_province_name="stateOrProvinceName",
        organization_name="organizationName",
        organization_unit_name="organizationUnitName",
        serial_number=0,
        validity_start_in_seconds=0,
        validity_end_in_seconds=10 * 365 * 24 * 60 * 60,
        out_key_file="private.key",
        out_cert_file="self-signed.crt",
        ca_cert_file="cert.crt",
        ca_key_file="key.key"):
    """ Creates a self_signed certificate

        Parameters:
            email_address: certificate email_address
            common_name: certificate common name
            country_name: certificate country name
            locality_name: certificate locality name
            state_or_province_name: certificate state/province name
            organization_name: certificate organization name
            organization_unit_name: certificate organization unit name
            serial_number: certificate serial number
            validity_start_in_seconds: certificate start validity in seconds
            validity_end_in_seconds: certificate end of validity in seconds
            out_key_file: output file for the certificate private key
            out_cert_file: output file for certificate
            ca_cert_file: certificate to be used to sign the generated certificate
            ca_key_file: key to be used to sign the generated certificate
    """
    # can look at generated file using openssl:
    # openssl x509 -inform pem -in selfsigned.crt -noout -text
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = country_name
    cert.get_subject().ST = state_or_province_name
    cert.get_subject().L = locality_name
    cert.get_subject().O = organization_name
    cert.get_subject().OU = organization_unit_name
    cert.get_subject().CN = common_name
    cert.get_subject().emailAddress = email_address
    cert.set_serial_number(serial_number)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(validity_end_in_seconds)
    cert.set_pubkey(k)

    st_key = open(ca_key_file, 'rt').read()
    st_cert = open(ca_cert_file, 'rt').read()

    ca_key = crypto.load_privatekey(crypto.FILETYPE_PEM, st_key)
    ca_cert = crypto.load_certificate(crypto.FILETYPE_PEM, st_cert)
    cert.set_issuer(ca_cert.get_subject())
    cert.sign(ca_key, 'sha512')

    with open(out_cert_file, "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
    with open(out_key_file, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))
